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


def _to_uint8(images: np.ndarray) -> np.ndarray:
    images = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    return images


def _make_grid(images: np.ndarray, columns: int, pad: int = 2) -> np.ndarray:
    images = _to_uint8(images)
    count, height, width, channels = images.shape
    rows = int(np.ceil(count / columns))
    grid = np.full(
        (
            rows * height + (rows + 1) * pad,
            columns * width + (columns + 1) * pad,
            channels,
        ),
        255,
        dtype=np.uint8,
    )

    for i, image in enumerate(images):
        row = i // columns
        col = i % columns
        y = pad + row * (height + pad)
        x = pad + col * (width + pad)
        grid[y : y + height, x : x + width] = image
    return grid


def write_ppm(path: str, image: np.ndarray) -> None:
    if image.shape[-1] != 3:
        raise ValueError("PPM export expects RGB images.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
    output.write_bytes(header + image.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic_debug.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--split", choices=("train", "hyperval", "test"), default="train")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-random-init", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_dataset(cfg["data"], seed=cfg["seed"])
    images = {
        "train": data.train_images,
        "hyperval": data.hyperval_images,
        "test": data.test_images,
    }[args.split][: args.num_images]

    aug_cfg = cfg["augnet"]
    input_shape = (1, *images.shape[1:])
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

    rng = jax.random.PRNGKey(args.seed + 1)
    augmented = augnet.apply(
        {"params": state.params},
        jnp.asarray(images),
        train=True,
        rngs={"dropout": rng},
    )

    interleaved = np.empty((images.shape[0] * 2, *images.shape[1:]), dtype=np.float32)
    interleaved[0::2] = images
    interleaved[1::2] = np.asarray(augmented)
    grid = _make_grid(interleaved, columns=2)

    output = args.output or str(Path(cfg["checkpoint_dir"]) / "augnet_samples.ppm")
    write_ppm(output, grid)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
