from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from data import load_dataset
from transformation_network import CIFARAugmentationNetwork, create_augnet_state
from utils import load_config, restore_state
from scripts.visualize_augnet import write_ppm


def _to_uint8(images: np.ndarray) -> np.ndarray:
    images = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    return images


def _base_grid(height: int, width: int) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([yy, xx], axis=-1)


def _flow_to_rgb(sample_grid: np.ndarray) -> np.ndarray:
    _, height, width, _ = sample_grid.shape
    base = _base_grid(height, width)[None, ...]
    flow = sample_grid - base
    max_abs = max(float(np.max(np.abs(flow))), 1e-6)
    magnitude = np.linalg.norm(flow, axis=-1)
    max_mag = max(float(np.max(magnitude)), 1e-6)
    rgb = np.stack(
        [
            0.5 + 0.5 * flow[..., 1] / max_abs,
            0.5 + 0.5 * flow[..., 0] / max_abs,
            magnitude / max_mag,
        ],
        axis=-1,
    )
    return np.clip(rgb, 0.0, 1.0)


def _centered_delta_to_rgb(delta: np.ndarray) -> np.ndarray:
    max_abs = max(float(np.max(np.abs(delta))), 1e-6)
    return np.clip(0.5 + 0.5 * delta / max_abs, 0.0, 1.0)


def _make_row_grid(rows: list[np.ndarray], pad: int = 2) -> np.ndarray:
    uint_rows = [_to_uint8(row) for row in rows]
    row_count = len(uint_rows)
    columns, height, width, channels = uint_rows[0].shape
    grid = np.full(
        (
            row_count * height + (row_count + 1) * pad,
            columns * width + (columns + 1) * pad,
            channels,
        ),
        255,
        dtype=np.uint8,
    )
    for row_index, row in enumerate(uint_rows):
        for col_index, image in enumerate(row):
            y = pad + row_index * (height + pad)
            x = pad + col_index * (width + pad)
            grid[y : y + height, x : x + width] = image
    return grid


def _select_split(data, split: str) -> np.ndarray:
    return {
        "train": data.train_images,
        "hyperval": data.hyperval_images,
        "test": data.test_images,
    }[split]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic_debug.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--split", choices=("train", "hyperval", "test"), default="train")
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-random-init", action="store_true")
    args = parser.parse_args()

    if args.steps < 2:
        raise ValueError("--steps must be at least 2.")

    cfg = load_config(args.config)
    data = load_dataset(cfg["data"], seed=cfg["seed"])
    images = _select_split(data, args.split)
    if args.source_index >= len(images) or args.target_index >= len(images):
        raise IndexError("source-index and target-index must be within the selected split.")

    source = images[args.source_index : args.source_index + 1]
    target = images[args.target_index : args.target_index + 1]

    aug_cfg = cfg["augnet"]
    input_shape = (1, *source.shape[1:])
    augnet = CIFARAugmentationNetwork(
        image_size=input_shape[1],
        channels=input_shape[-1],
        tau_dim=aug_cfg["tau_dim"],
        tau_dropout=aug_cfg["tau_dropout"],
        use_appearance=aug_cfg.get("use_appearance", True),
        encoder_widths=tuple(aug_cfg.get("encoder_widths", (16, 32, 64, 128))),
        decoder_widths=tuple(aug_cfg.get("decoder_widths", (64, 32, 16))),
        decoder_base_width=aug_cfg.get("decoder_base_width", 128),
    )
    state = create_augnet_state(
        jax.random.PRNGKey(args.seed),
        augnet,
        input_shape=input_shape,
        learning_rate=aug_cfg["learning_rate"],
    )
    checkpoint = args.checkpoint or str(Path(cfg["checkpoint_dir"]) / "augnet.msgpack")
    if Path(checkpoint).exists():
        state = restore_state(checkpoint, state)
    elif not args.allow_random_init:
        raise FileNotFoundError(
            f"AugNet checkpoint not found: {checkpoint}. "
            "Pass --checkpoint explicitly or use --allow-random-init for an untrained preview."
        )

    _, source_aux = augnet.apply(
        {"params": state.params},
        jnp.asarray(source),
        train=False,
        return_aux=True,
    )
    _, target_aux = augnet.apply(
        {"params": state.params},
        jnp.asarray(target),
        train=False,
        return_aux=True,
    )

    alphas = jnp.linspace(0.0, 1.0, args.steps)[:, None]
    tau = (1.0 - alphas) * source_aux["tau"] + alphas * target_aux["tau"]
    source_tiled = jnp.repeat(jnp.asarray(source), args.steps, axis=0)
    augmented, aux = augnet.apply(
        {"params": state.params},
        source_tiled,
        train=False,
        return_aux=True,
        tau_override=tau,
    )

    original_row = np.repeat(source, args.steps, axis=0)
    flow_row = _flow_to_rgb(np.asarray(aux["sample_grid"]))
    spatial_row = np.asarray(aux["spatial_images"])
    appearance_delta = np.asarray(aux.get("appearance_delta", jnp.zeros_like(augmented)))
    appearance_row = _centered_delta_to_rgb(appearance_delta)
    final_row = np.asarray(augmented)

    grid = _make_row_grid(
        [
            original_row,
            flow_row,
            spatial_row,
            appearance_row,
            final_row,
        ]
    )
    output = args.output or str(Path(cfg["checkpoint_dir"]) / "tau_interpolation.ppm")
    write_ppm(output, grid)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
