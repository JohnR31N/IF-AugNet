from pathlib import Path
import argparse
import sys
from typing import Dict, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
import optax

from classification_network import (
    ImageNetResNet,
    MnistConvNet,
    PyramidNet272ShakeDrop,
    PyramidNetShakeDrop,
    ResNet18,
    ResNet50,
    ResNet200,
    ResNet56,
    ShakeShake26x2x32d,
    ShakeShakeResNet,
    WideResNet,
    WideResNet28x10,
    classifier_eval_step,
    classifier_train_step,
    classifier_train_step_with_augnet,
    create_classifier_state,
)
from data import NumpyBatchIterator, load_dataset
from paramyield_network import (
    augnet_influence_train_step,
    compute_batch_s_test,
    compute_batch_s_test_residual,
)
from transformation_network import (
    CIFARAugmentationNetwork,
    FeatureDiscriminator,
    ImageDiscriminator,
    augnet_pretrain_step,
    create_augnet_state,
    create_discriminator_state,
)
from utils import (
    JsonlMetricLogger,
    load_pickle,
    load_config,
    restore_state,
    save_pickle,
    save_state,
    validate_config,
    write_run_manifest,
)


def _to_float(metrics: Dict[str, jnp.ndarray]) -> Dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    values = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    return f"{prefix} {values}"


def build_classifier(config: Dict) -> object:
    backbone = config.get("backbone", "resnet56")
    if backbone in ("mnist_cnn", "4layer_cnn", "mnist_4layer_cnn"):
        return MnistConvNet(
            num_classes=config["num_classes"],
            widths=tuple(config.get("widths", (32, 32, 64, 64))),
            feature_dim=config.get("feature_dim", 128),
        )
    if backbone == "resnet50":
        return ResNet50(num_classes=config["num_classes"])
    if backbone == "resnet200":
        return ResNet200(num_classes=config["num_classes"])
    if backbone == "imagenet_resnet":
        return ImageNetResNet(
            stage_sizes=tuple(config.get("stage_sizes", (3, 4, 6, 3))),
            widths=tuple(config.get("widths", (64, 128, 256, 512))),
            stem_width=config.get("stem_width", 64),
            num_classes=config["num_classes"],
        )
    if backbone == "resnet18":
        return ResNet18(
            num_classes=config["num_classes"],
            width_multiplier=config["width_multiplier"],
        )
    if backbone == "resnet56":
        return ResNet56(
            num_classes=config["num_classes"],
            width_multiplier=config["width_multiplier"],
        )
    if backbone in ("wide_resnet28_10", "wrn28_10"):
        return WideResNet28x10(
            num_classes=config["num_classes"],
            dropout_rate=config.get("dropout_rate", 0.0),
        )
    if backbone == "wide_resnet":
        return WideResNet(
            depth=config.get("depth", 28),
            width_multiplier=config.get("width_multiplier", 10),
            num_classes=config["num_classes"],
            dropout_rate=config.get("dropout_rate", 0.0),
        )
    if backbone in ("shakeshake26_2x32d", "shake_shake26_2x32d"):
        return ShakeShake26x2x32d(num_classes=config["num_classes"])
    if backbone in ("shakeshake26_2x96d", "shake_shake26_2x96d"):
        return ShakeShakeResNet(
            depth=26,
            base_width=96,
            num_classes=config["num_classes"],
        )
    if backbone == "shake_shake":
        return ShakeShakeResNet(
            depth=config.get("depth", 26),
            base_width=config.get("base_width", 32),
            num_classes=config["num_classes"],
        )
    if backbone in ("pyramidnet272_shakedrop", "pyramidnet_shakedrop272"):
        return PyramidNet272ShakeDrop(
            num_classes=config["num_classes"],
            final_keep_prob=config.get("final_keep_prob", 0.5),
        )
    if backbone == "pyramidnet_shakedrop":
        return PyramidNetShakeDrop(
            depth=config.get("depth", 272),
            alpha=config.get("alpha", 200),
            num_classes=config["num_classes"],
            bottleneck=config.get("bottleneck", True),
            final_keep_prob=config.get("final_keep_prob", 0.5),
        )
    raise ValueError(f"Unknown classifier backbone: {backbone}")


def infer_input_shape(splits) -> tuple[int, int, int, int]:
    return (1, *splits.train_images.shape[1:])


def infer_feature_dim(state, model, input_shape) -> int:
    features, _ = model.apply(
        {"params": state.params, "batch_stats": state.batch_stats},
        jnp.ones(input_shape, jnp.float32),
        train=False,
        return_features=True,
    )
    return int(features.shape[-1])


def build_augnet(config: Dict, input_shape: tuple[int, int, int, int]) -> CIFARAugmentationNetwork:
    aug_cfg = config["augnet"]
    return CIFARAugmentationNetwork(
        image_size=input_shape[1],
        channels=input_shape[-1],
        tau_dim=aug_cfg["tau_dim"],
        tau_dropout=aug_cfg["tau_dropout"],
        spatial_scale=aug_cfg.get("spatial_scale", 0.20),
        appearance_scale=aug_cfg.get("appearance_scale", 0.25),
        smoothing_kernel=aug_cfg.get("smoothing_kernel", 4),
        use_appearance=aug_cfg.get("use_appearance", True),
        encoder_widths=tuple(aug_cfg.get("encoder_widths", (16, 32, 64, 128))),
        decoder_widths=tuple(aug_cfg.get("decoder_widths", (64, 32, 16))),
        decoder_base_width=aug_cfg.get("decoder_base_width", 128),
    )


def create_augnet_state_from_config(
    rng,
    augnet,
    input_shape: tuple[int, int, int, int],
    config: Dict,
) -> object:
    return create_augnet_state(
        rng,
        augnet,
        input_shape=input_shape,
        learning_rate=config["learning_rate"],
        optimizer=config.get("optimizer", "adam"),
        adam_beta1=config.get("adam_beta1", 0.9),
        adam_beta2=config.get("adam_beta2", 0.999),
        gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
        zero_nonfinite_grads=config.get("zero_nonfinite_grads", True),
    )


def pretrain_augnet_optimizer_config(aug_cfg: Dict, pretrain_cfg: Dict) -> Dict:
    config = dict(aug_cfg)
    config["learning_rate"] = pretrain_cfg.get("augnet_learning_rate", pretrain_cfg["learning_rate"])
    config["optimizer"] = pretrain_cfg.get("augnet_optimizer", aug_cfg.get("optimizer", "adam"))
    config["adam_beta1"] = pretrain_cfg.get(
        "augnet_adam_beta1",
        pretrain_cfg.get("beta1", aug_cfg.get("adam_beta1", 0.9)),
    )
    config["adam_beta2"] = pretrain_cfg.get(
        "augnet_adam_beta2",
        pretrain_cfg.get("beta2", aug_cfg.get("adam_beta2", 0.999)),
    )
    config["gradient_clip_norm"] = pretrain_cfg.get(
        "augnet_gradient_clip_norm",
        aug_cfg.get("gradient_clip_norm", 1.0),
    )
    config["zero_nonfinite_grads"] = pretrain_cfg.get(
        "augnet_zero_nonfinite_grads",
        aug_cfg.get("zero_nonfinite_grads", True),
    )
    return config


def build_learning_rate(config: Dict, steps_per_epoch: int):
    learning_rate = config["learning_rate"]
    decay_epochs = config.get("lr_decay_epochs", [])
    if not decay_epochs:
        return learning_rate

    decay_factor = config.get("lr_decay_factor", 0.1)
    boundaries_and_scales = {
        int(epoch * steps_per_epoch): decay_factor for epoch in decay_epochs
    }
    return optax.piecewise_constant_schedule(
        init_value=learning_rate,
        boundaries_and_scales=boundaries_and_scales,
    )


def steps_per_epoch(num_examples: int, batch_size: int, drop_last: bool = False) -> int:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if drop_last:
        return max(1, num_examples // batch_size)
    return max(1, (num_examples + batch_size - 1) // batch_size)


def progressive_image_size_for_step(config: Dict, step: int, default_image_size: int) -> int | None:
    sizes = config.get("progressive_image_sizes")
    if not sizes:
        return None
    boundaries = config.get("progressive_boundaries", [])
    index = 0
    for boundary in boundaries:
        if step >= int(boundary):
            index += 1
    index = min(index, len(sizes) - 1)
    image_size = int(sizes[index])
    if image_size <= 0 or image_size >= default_image_size:
        return None
    return image_size


def tree_average(trees):
    if not trees:
        raise ValueError("Cannot average an empty list of pytrees.")
    inv_count = 1.0 / len(trees)
    return jax.tree_util.tree_map(lambda *xs: sum(xs) * inv_count, *trees)


def build_normalization(data_config: Dict):
    if not data_config.get("normalize", False):
        return None, None
    channels = len(data_config["mean"])
    mean = jnp.asarray(data_config["mean"], dtype=jnp.float32).reshape((1, 1, 1, channels))
    std = jnp.asarray(data_config["std"], dtype=jnp.float32).reshape((1, 1, 1, channels))
    return mean, std


def rng_to_list(rng) -> list:
    return np.asarray(rng).tolist()


def rng_from_list(values) -> jax.Array:
    return jnp.asarray(values, dtype=jnp.uint32)


def stage_paths(checkpoint_dir: Path, stage: str) -> Dict[str, Path]:
    return {
        "state": checkpoint_dir / f"{stage}_latest.msgpack",
        "progress": checkpoint_dir / f"{stage}_progress.pkl",
    }


def stage_completed(checkpoint_dir: Path, stage: str) -> bool:
    progress = stage_paths(checkpoint_dir, stage)["progress"]
    if not progress.exists():
        return False
    return bool(load_pickle(str(progress)).get("completed", False))


def save_stage_progress(
    checkpoint_dir: Path,
    stage: str,
    state,
    iterator: NumpyBatchIterator,
    rng,
    next_step: int,
    total_steps: int,
    completed: bool = False,
) -> None:
    paths = stage_paths(checkpoint_dir, stage)
    save_state(str(paths["state"]), state)
    save_pickle(
        str(paths["progress"]),
        {
            "completed": completed,
            "iterator": iterator.state_dict(),
            "next_step": next_step,
            "rng": rng_to_list(rng),
            "total_steps": total_steps,
        },
    )


def restore_stage_progress(
    checkpoint_dir: Path,
    stage: str,
    state,
    iterator: NumpyBatchIterator,
):
    paths = stage_paths(checkpoint_dir, stage)
    progress = load_pickle(str(paths["progress"]))
    state = restore_state(str(paths["state"]), state)
    iterator.load_state_dict(progress["iterator"])
    return state, iterator, rng_from_list(progress["rng"]), int(progress["next_step"])


def should_checkpoint(step: int, every: int) -> bool:
    return every > 0 and (step + 1) % every == 0


def stop_step_for_stage(start_step: int, total_steps: int, stop_after_steps: int | None) -> int:
    if stop_after_steps is None:
        return total_steps
    if stop_after_steps <= 0:
        raise ValueError(f"stop_after_steps must be positive, got {stop_after_steps}.")
    return min(total_steps, start_step + stop_after_steps)


def evaluate(
    state,
    model,
    iterator: Iterable[Dict[str, jnp.ndarray]],
    batches: int,
    image_mean=None,
    image_std=None,
) -> Dict[str, float]:
    totals = None
    total_examples = 0
    for _ in range(batches):
        batch = next(iterator)
        batch_examples = int(batch["label"].shape[0])
        metrics = _to_float(
            classifier_eval_step(
                state,
                model,
                batch,
                image_mean=image_mean,
                image_std=image_std,
            )
        )
        if totals is None:
            totals = {key: 0.0 for key in metrics}
        for key, value in metrics.items():
            totals[key] += value * batch_examples
        total_examples += batch_examples
    if totals is None or total_examples == 0:
        raise ValueError("No evaluation examples were produced.")
    return {key: value / total_examples for key, value in totals.items()}


def _normalize_for_model(images, image_mean=None, image_std=None):
    images = jnp.asarray(images)
    if image_mean is None or image_std is None:
        return images
    return (images - image_mean) / image_std


def _sanitize_recalibrated_batch_stats(batch_stats):
    def clean(value):
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            return value
        value = jnp.nan_to_num(value, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        return jnp.clip(value, -1.0e4, 1.0e4)

    return jax.tree_util.tree_map(clean, batch_stats)


def recalibrate_batch_stats(
    state,
    model,
    iterator: Iterable[Dict[str, jnp.ndarray]],
    batches: int,
    image_mean=None,
    image_std=None,
):
    if state.batch_stats is None or batches <= 0:
        return state
    batch_stats = state.batch_stats
    for _ in range(batches):
        batch = next(iterator)
        variables = {"params": state.params, "batch_stats": batch_stats}
        _, updates = model.apply(
            variables,
            _normalize_for_model(batch["image"], image_mean, image_std),
            train=True,
            mutable=["batch_stats"],
            rngs={"dropout": jax.random.PRNGKey(0)},
        )
        batch_stats = _sanitize_recalibrated_batch_stats(updates["batch_stats"])
    return state.replace(batch_stats=batch_stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10.yaml")
    parser.add_argument(
        "--stage",
        choices=("classifier", "pretrain_augnet", "augnet", "retrain", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        default=None,
        help="Run at most this many update steps in a single selected stage, save progress, and exit.",
    )
    args = parser.parse_args()
    if args.stop_after_steps is not None and args.stage == "all":
        parser.error("--stop-after-steps requires selecting one stage, not --stage all.")

    cfg = load_config(args.config)
    validate_config(cfg)
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_every_steps = cfg.get("checkpoint_every_steps", 0)
    classifier_ckpt = checkpoint_dir / "classifier.msgpack"
    augnet_ckpt = checkpoint_dir / "augnet.msgpack"
    retrained_classifier_ckpt = checkpoint_dir / "classifier_retrained.msgpack"
    metric_logger = JsonlMetricLogger(str(checkpoint_dir / "metrics.jsonl"))
    write_run_manifest(str(checkpoint_dir / "run_manifest.json"), cfg, " ".join(sys.argv))

    rng = jax.random.PRNGKey(cfg["seed"])
    rng_model, rng_aug, rng_image_d, rng_feature_d, rng_retrain = jax.random.split(rng, 5)

    data_cfg = cfg["data"]
    splits = load_dataset(data_cfg, seed=cfg["seed"])
    input_shape = infer_input_shape(splits)
    image_size = input_shape[1]
    channels = input_shape[-1]
    image_mean, image_std = build_normalization(data_cfg)

    classifier_cfg = cfg["classifier"]
    classifier = build_classifier(classifier_cfg)
    classifier_steps_per_epoch = steps_per_epoch(
        len(splits.train_images),
        classifier_cfg["batch_size"],
        drop_last=classifier_cfg.get("drop_last", False),
    )
    classifier_state = create_classifier_state(
        rng_model,
        classifier,
        input_shape=input_shape,
        learning_rate=build_learning_rate(classifier_cfg, classifier_steps_per_epoch),
        optimizer=classifier_cfg["optimizer"],
        momentum=classifier_cfg["momentum"],
        weight_decay=classifier_cfg["weight_decay"],
        gradient_clip_norm=classifier_cfg.get("gradient_clip_norm", 0.0),
        zero_nonfinite_grads=classifier_cfg.get("zero_nonfinite_grads", False),
    )

    aug_cfg = cfg["augnet"]
    augnet = build_augnet(cfg, input_shape)
    aug_state = create_augnet_state_from_config(rng_aug, augnet, input_shape, aug_cfg)

    train_iter = NumpyBatchIterator(
        splits.train_images,
        splits.train_labels,
        classifier_cfg["batch_size"],
        seed=cfg["seed"],
        drop_last=classifier_cfg.get("drop_last", False),
    )
    test_iter = NumpyBatchIterator(
        splits.test_images,
        splits.test_labels,
        classifier_cfg["batch_size"],
        seed=cfg["seed"],
        shuffle=False,
        drop_last=False,
    )

    if args.stage in ("pretrain_augnet", "augnet"):
        classifier_state = restore_state(str(classifier_ckpt), classifier_state)

    if args.stage == "retrain" and augnet_ckpt.exists():
        aug_state = restore_state(str(augnet_ckpt), aug_state, restore_opt_state=False)

    if args.stage in ("classifier", "all"):
        classifier_total_steps = classifier_cfg["epochs"] * classifier_steps_per_epoch
        classifier_start_step = 0
        if args.resume and stage_completed(checkpoint_dir, "classifier"):
            classifier_state = restore_state(str(classifier_ckpt), classifier_state)
            print("classifier stage already complete; restored final checkpoint")
            metric_logger.log("resume_classifier", {"skipped": 1.0})
        else:
            if args.resume and stage_paths(checkpoint_dir, "classifier")["progress"].exists():
                classifier_state, train_iter, rng, classifier_start_step = restore_stage_progress(
                    checkpoint_dir,
                    "classifier",
                    classifier_state,
                    train_iter,
                )
                metric_logger.log(
                    "resume_classifier",
                    {"next_step": float(classifier_start_step)},
                )

            classifier_stop_step = stop_step_for_stage(
                classifier_start_step,
                classifier_total_steps,
                args.stop_after_steps,
            )
            progress = trange(
                classifier_start_step,
                classifier_stop_step,
                desc="classifier",
            )
            for global_step in progress:
                epoch = global_step // classifier_steps_per_epoch
                step = global_step % classifier_steps_per_epoch
                rng, step_rng = jax.random.split(rng)
                classifier_state, metrics = classifier_train_step(
                    classifier_state,
                    classifier,
                    next(train_iter),
                    step_rng,
                    apply_baseline_augmentation=classifier_cfg.get("baseline_augmentation", True),
                    cutout_size=classifier_cfg["cutout_size"],
                    image_mean=image_mean,
                    image_std=image_std,
                )
                if global_step % cfg["log_every"] == 0:
                    metrics_float = _to_float(metrics)
                    progress.set_postfix(metrics_float)
                    metric_logger.log(
                        "classifier_train",
                        metrics_float,
                        step=global_step,
                        epoch=epoch + 1,
                    )
                if should_checkpoint(global_step, checkpoint_every_steps):
                    save_stage_progress(
                        checkpoint_dir,
                        "classifier",
                        classifier_state,
                        train_iter,
                        rng,
                        global_step + 1,
                        classifier_total_steps,
                    )

            if classifier_stop_step < classifier_total_steps:
                save_stage_progress(
                    checkpoint_dir,
                    "classifier",
                    classifier_state,
                    train_iter,
                    rng,
                    classifier_stop_step,
                    classifier_total_steps,
                )
                print(
                    f"classifier stage paused at "
                    f"{classifier_stop_step}/{classifier_total_steps} steps"
                )
            else:
                eval_metrics = evaluate(
                    classifier_state,
                    classifier,
                    test_iter,
                    cfg["eval_batches"],
                    image_mean=image_mean,
                    image_std=image_std,
                )
                print(_format_metrics("classifier_eval", eval_metrics))
                metric_logger.log("classifier_eval", eval_metrics)
                save_state(str(classifier_ckpt), classifier_state)
                save_stage_progress(
                    checkpoint_dir,
                    "classifier",
                    classifier_state,
                    train_iter,
                    rng,
                    classifier_total_steps,
                    classifier_total_steps,
                    completed=True,
                )

    if args.stage in ("pretrain_augnet", "all"):
        pretrain_cfg = cfg["pretrain"]
        aug_state = create_augnet_state_from_config(
            rng_aug,
            augnet,
            input_shape,
            pretrain_augnet_optimizer_config(aug_cfg, pretrain_cfg),
        )
        image_discriminator = ImageDiscriminator()
        feature_discriminator = FeatureDiscriminator()
        feature_dim = infer_feature_dim(classifier_state, classifier, input_shape)
        image_discriminator_state = create_discriminator_state(
            rng_image_d,
            image_discriminator,
            input_shape=input_shape,
            learning_rate=pretrain_cfg["learning_rate"],
            beta1=pretrain_cfg["beta1"],
            beta2=pretrain_cfg["beta2"],
        )
        feature_discriminator_state = create_discriminator_state(
            rng_feature_d,
            feature_discriminator,
            input_shape=(1, feature_dim),
            learning_rate=pretrain_cfg["learning_rate"],
            beta1=pretrain_cfg["beta1"],
            beta2=pretrain_cfg["beta2"],
        )
        pretrain_iter = NumpyBatchIterator(
            splits.train_images,
            splits.train_labels,
            pretrain_cfg["batch_size"],
            seed=cfg["seed"] + 3,
        )
        pretrain_image_d_latest = checkpoint_dir / "pretrain_augnet_image_discriminator_latest.msgpack"
        pretrain_feature_d_latest = checkpoint_dir / "pretrain_augnet_feature_discriminator_latest.msgpack"

        pretrain_start_step = 0
        if args.resume and stage_completed(checkpoint_dir, "pretrain_augnet"):
            aug_state = restore_state(str(augnet_ckpt), aug_state)
            print("pretrain_augnet stage already complete; restored final AugNet checkpoint")
            metric_logger.log("resume_pretrain_augnet", {"skipped": 1.0})
        else:
            if args.resume and stage_paths(checkpoint_dir, "pretrain_augnet")["progress"].exists():
                aug_state, pretrain_iter, rng, pretrain_start_step = restore_stage_progress(
                    checkpoint_dir,
                    "pretrain_augnet",
                    aug_state,
                    pretrain_iter,
                )
                image_discriminator_state = restore_state(
                    str(pretrain_image_d_latest),
                    image_discriminator_state,
                )
                feature_discriminator_state = restore_state(
                    str(pretrain_feature_d_latest),
                    feature_discriminator_state,
                )
                metric_logger.log(
                    "resume_pretrain_augnet",
                    {"next_step": float(pretrain_start_step)},
                )

            pretrain_stop_step = stop_step_for_stage(
                pretrain_start_step,
                pretrain_cfg["steps"],
                args.stop_after_steps,
            )
            progress = trange(pretrain_start_step, pretrain_stop_step, desc="pretrain_augnet")
            for step in progress:
                rng, pretrain_rng = jax.random.split(rng)
                (
                    aug_state,
                    image_discriminator_state,
                    feature_discriminator_state,
                    metrics,
                ) = augnet_pretrain_step(
                    aug_state,
                    augnet,
                    image_discriminator_state,
                    image_discriminator,
                    feature_discriminator_state,
                    feature_discriminator,
                    classifier_state,
                    classifier,
                    next(pretrain_iter),
                    pretrain_rng,
                    apply_baseline_augmentation=pretrain_cfg.get("baseline_augmentation", True),
                    cutout_size=pretrain_cfg["cutout_size"],
                    progressive_image_size=progressive_image_size_for_step(
                        pretrain_cfg,
                        step,
                        image_size,
                    ),
                    image_loss_weight=pretrain_cfg["image_loss_weight"],
                    feature_loss_weight=pretrain_cfg["feature_loss_weight"],
                    identity_l2_weight=pretrain_cfg["identity_l2_weight"],
                    image_mean=image_mean,
                    image_std=image_std,
                )
                if step % cfg["log_every"] == 0:
                    metrics_float = _to_float(metrics)
                    progress.set_postfix(metrics_float)
                    metric_logger.log("pretrain_augnet", metrics_float, step=step)
                if should_checkpoint(step, checkpoint_every_steps):
                    save_stage_progress(
                        checkpoint_dir,
                        "pretrain_augnet",
                        aug_state,
                        pretrain_iter,
                        rng,
                        step + 1,
                        pretrain_cfg["steps"],
                    )
                    save_state(str(pretrain_image_d_latest), image_discriminator_state)
                    save_state(str(pretrain_feature_d_latest), feature_discriminator_state)

            if pretrain_stop_step < pretrain_cfg["steps"]:
                save_stage_progress(
                    checkpoint_dir,
                    "pretrain_augnet",
                    aug_state,
                    pretrain_iter,
                    rng,
                    pretrain_stop_step,
                    pretrain_cfg["steps"],
                )
                save_state(str(pretrain_image_d_latest), image_discriminator_state)
                save_state(str(pretrain_feature_d_latest), feature_discriminator_state)
                print(
                    f"pretrain_augnet stage paused at "
                    f"{pretrain_stop_step}/{pretrain_cfg['steps']} steps"
                )
            else:
                metrics_float = _to_float(metrics)
                print(_format_metrics("pretrain_augnet_last", metrics_float))
                metric_logger.log("pretrain_augnet_last", metrics_float)
                save_state(str(augnet_ckpt), aug_state)
                save_stage_progress(
                    checkpoint_dir,
                    "pretrain_augnet",
                    aug_state,
                    pretrain_iter,
                    rng,
                    pretrain_cfg["steps"],
                    pretrain_cfg["steps"],
                    completed=True,
                )
                save_state(str(pretrain_image_d_latest), image_discriminator_state)
                save_state(str(pretrain_feature_d_latest), feature_discriminator_state)

    if args.stage in ("augnet", "all"):
        aug_state = create_augnet_state_from_config(rng_aug, augnet, input_shape, aug_cfg)
        if augnet_ckpt.exists():
            aug_state = restore_state(str(augnet_ckpt), aug_state, restore_opt_state=False)

        aug_train_iter = NumpyBatchIterator(
            splits.train_images,
            splits.train_labels,
            aug_cfg["batch_size"],
            seed=cfg["seed"] + 1,
        )
        hyperval_iter = NumpyBatchIterator(
            splits.hyperval_images,
            splits.hyperval_labels,
            aug_cfg["batch_size"],
            seed=cfg["seed"] + 2,
        )

        augnet_start_step = 0
        if args.resume and stage_completed(checkpoint_dir, "augnet"):
            aug_state = restore_state(str(augnet_ckpt), aug_state, restore_opt_state=False)
            print("augnet stage already complete; restored final AugNet checkpoint")
            metric_logger.log("resume_augnet", {"skipped": 1.0})
        else:
            fixed_s_test = None
            if aug_cfg.get("precompute_s_test", True):
                s_test_train_iter = NumpyBatchIterator(
                    splits.train_images,
                    splits.train_labels,
                    aug_cfg["batch_size"],
                    seed=cfg["seed"] + 3,
                )
                s_test_hyperval_iter = NumpyBatchIterator(
                    splits.hyperval_images,
                    splits.hyperval_labels,
                    aug_cfg["batch_size"],
                    seed=cfg["seed"] + 4,
                )
                s_test_batches = aug_cfg.get("s_test_batches", 1)
                s_tests = []
                s_test_residuals = []
                progress = trange(s_test_batches, desc="precompute_s_test")
                for _ in progress:
                    train_batch = next(s_test_train_iter)
                    val_batch = next(s_test_hyperval_iter)
                    s_test = compute_batch_s_test(
                        classifier_state,
                        classifier,
                        train_batch,
                        val_batch,
                        damping=aug_cfg["damping"],
                        cg_iters=aug_cfg["cg_iters"],
                        image_mean=image_mean,
                        image_std=image_std,
                    )
                    s_tests.append(s_test)
                    s_test_residuals.append(
                        float(
                            compute_batch_s_test_residual(
                                classifier_state,
                                classifier,
                                train_batch,
                                val_batch,
                                s_test,
                                damping=aug_cfg["damping"],
                                image_mean=image_mean,
                                image_std=image_std,
                            )
                        )
                    )
                fixed_s_test = tree_average(s_tests)
                metric_logger.log(
                    "precompute_s_test",
                    {
                        "batches": float(s_test_batches),
                        "damping": float(aug_cfg["damping"]),
                        "cg_iters": float(aug_cfg["cg_iters"]),
                        "residual_mean": float(np.mean(s_test_residuals)),
                        "residual_max": float(np.max(s_test_residuals)),
                    },
                )

            if args.resume and stage_paths(checkpoint_dir, "augnet")["progress"].exists():
                aug_state, aug_train_iter, rng, augnet_start_step = restore_stage_progress(
                    checkpoint_dir,
                    "augnet",
                    aug_state,
                    aug_train_iter,
                )
                metric_logger.log(
                    "resume_augnet",
                    {"next_step": float(augnet_start_step)},
                )

            augnet_stop_step = stop_step_for_stage(
                augnet_start_step,
                aug_cfg["steps"],
                args.stop_after_steps,
            )
            progress = trange(augnet_start_step, augnet_stop_step, desc="augnet")
            for step in progress:
                rng, aug_rng = jax.random.split(rng)
                train_batch = next(aug_train_iter)
                if fixed_s_test is None:
                    val_batch = next(hyperval_iter)
                    s_test = compute_batch_s_test(
                        classifier_state,
                        classifier,
                        train_batch,
                        val_batch,
                        damping=aug_cfg["damping"],
                        cg_iters=aug_cfg["cg_iters"],
                        image_mean=image_mean,
                        image_std=image_std,
                    )
                else:
                    s_test = fixed_s_test
                aug_state, metrics = augnet_influence_train_step(
                    aug_state,
                    augnet,
                    classifier_state,
                    classifier,
                    train_batch,
                    s_test,
                    aug_rng,
                    identity_l2_weight=aug_cfg["identity_l2_weight"],
                    influence_clip_value=aug_cfg.get("influence_clip_value", 0.0),
                    label_preservation_weight=aug_cfg.get("label_preservation_weight", 0.0),
                    image_mean=image_mean,
                    image_std=image_std,
                )
                if step % cfg["log_every"] == 0:
                    metrics_float = _to_float(metrics)
                    progress.set_postfix(metrics_float)
                    metric_logger.log("augnet", metrics_float, step=step)
                if should_checkpoint(step, checkpoint_every_steps):
                    save_stage_progress(
                        checkpoint_dir,
                        "augnet",
                        aug_state,
                        aug_train_iter,
                        rng,
                        step + 1,
                        aug_cfg["steps"],
                    )

            if augnet_stop_step < aug_cfg["steps"]:
                save_stage_progress(
                    checkpoint_dir,
                    "augnet",
                    aug_state,
                    aug_train_iter,
                    rng,
                    augnet_stop_step,
                    aug_cfg["steps"],
                )
                print(
                    f"augnet stage paused at "
                    f"{augnet_stop_step}/{aug_cfg['steps']} steps"
                )
            else:
                metrics_float = _to_float(metrics)
                print(_format_metrics("augnet_last", metrics_float))
                metric_logger.log("augnet_last", metrics_float)
                save_state(str(augnet_ckpt), aug_state)
                save_stage_progress(
                    checkpoint_dir,
                    "augnet",
                    aug_state,
                    aug_train_iter,
                    rng,
                    aug_cfg["steps"],
                    aug_cfg["steps"],
                    completed=True,
                )

    if args.stage in ("retrain", "all"):
        if args.stage == "retrain":
            aug_state = restore_state(str(augnet_ckpt), aug_state, restore_opt_state=False)

        retrain_cfg = cfg["retrain"]
        retrain_batch_size = retrain_cfg.get("batch_size", classifier_cfg["batch_size"])
        retrain_steps_per_epoch = steps_per_epoch(
            len(splits.train_images),
            retrain_batch_size,
            drop_last=retrain_cfg.get("drop_last", False),
        )
        retrained_state = create_classifier_state(
            rng_retrain,
            classifier,
            input_shape=input_shape,
            learning_rate=build_learning_rate(retrain_cfg, retrain_steps_per_epoch),
            optimizer=retrain_cfg["optimizer"],
            momentum=retrain_cfg["momentum"],
            weight_decay=retrain_cfg["weight_decay"],
            gradient_clip_norm=retrain_cfg.get("gradient_clip_norm", 0.0),
            zero_nonfinite_grads=retrain_cfg.get("zero_nonfinite_grads", False),
        )
        retrain_iter = NumpyBatchIterator(
            splits.train_images,
            splits.train_labels,
            retrain_batch_size,
            seed=cfg["seed"] + 4,
            drop_last=retrain_cfg.get("drop_last", False),
        )

        retrain_total_steps = retrain_cfg["epochs"] * retrain_steps_per_epoch
        retrain_start_step = 0
        if args.resume and stage_completed(checkpoint_dir, "retrain"):
            retrained_state = restore_state(str(retrained_classifier_ckpt), retrained_state)
            print("retrain stage already complete; restored final checkpoint")
            metric_logger.log("resume_retrain", {"skipped": 1.0})
        else:
            if args.resume and stage_paths(checkpoint_dir, "retrain")["progress"].exists():
                retrained_state, retrain_iter, rng, retrain_start_step = restore_stage_progress(
                    checkpoint_dir,
                    "retrain",
                    retrained_state,
                    retrain_iter,
                )
                metric_logger.log(
                    "resume_retrain",
                    {"next_step": float(retrain_start_step)},
                )

            retrain_stop_step = stop_step_for_stage(
                retrain_start_step,
                retrain_total_steps,
                args.stop_after_steps,
            )
            progress = trange(retrain_start_step, retrain_stop_step, desc="retrain")
            for global_step in progress:
                epoch = global_step // retrain_steps_per_epoch
                step = global_step % retrain_steps_per_epoch
                rng, step_rng = jax.random.split(rng)
                retrained_state, metrics = classifier_train_step_with_augnet(
                    retrained_state,
                    classifier,
                    aug_state,
                    augnet,
                    next(retrain_iter),
                    step_rng,
                    apply_baseline_augmentation=retrain_cfg.get("baseline_augmentation", True),
                    cutout_size=retrain_cfg["cutout_size"],
                    learned_aug_probability=retrain_cfg.get("learned_aug_probability", 1.0),
                    learned_aug_input=retrain_cfg.get("learned_aug_input", "baseline"),
                    image_mean=image_mean,
                    image_std=image_std,
                )
                if global_step % cfg["log_every"] == 0:
                    metrics_float = _to_float(metrics)
                    progress.set_postfix(metrics_float)
                    metric_logger.log(
                        "retrain",
                        metrics_float,
                        step=global_step,
                        epoch=epoch + 1,
                    )
                if should_checkpoint(global_step, checkpoint_every_steps):
                    save_stage_progress(
                        checkpoint_dir,
                        "retrain",
                        retrained_state,
                        retrain_iter,
                        rng,
                        global_step + 1,
                        retrain_total_steps,
                    )

            if retrain_stop_step < retrain_total_steps:
                save_stage_progress(
                    checkpoint_dir,
                    "retrain",
                    retrained_state,
                    retrain_iter,
                    rng,
                    retrain_stop_step,
                    retrain_total_steps,
                )
                print(
                    f"retrain stage paused at "
                    f"{retrain_stop_step}/{retrain_total_steps} steps"
                )
            else:
                recalibration_batches = retrain_cfg.get("recalibrate_batch_stats_batches", 0)
                if recalibration_batches:
                    recalibration_iter = NumpyBatchIterator(
                        splits.train_images,
                        splits.train_labels,
                        retrain_batch_size,
                        seed=cfg["seed"] + 6,
                        shuffle=True,
                        drop_last=False,
                    )
                    retrained_state = recalibrate_batch_stats(
                        retrained_state,
                        classifier,
                        recalibration_iter,
                        int(recalibration_batches),
                        image_mean=image_mean,
                        image_std=image_std,
                    )
                test_iter = NumpyBatchIterator(
                    splits.test_images,
                    splits.test_labels,
                    classifier_cfg["batch_size"],
                    seed=cfg["seed"],
                    shuffle=False,
                    drop_last=False,
                )
                eval_metrics = evaluate(
                    retrained_state,
                    classifier,
                    test_iter,
                    cfg["eval_batches"],
                    image_mean=image_mean,
                    image_std=image_std,
                )
                print(_format_metrics("retrained_classifier_eval", eval_metrics))
                metric_logger.log("retrained_classifier_eval", eval_metrics)
                save_state(str(retrained_classifier_ckpt), retrained_state)
                save_stage_progress(
                    checkpoint_dir,
                    "retrain",
                    retrained_state,
                    retrain_iter,
                    rng,
                    retrain_total_steps,
                    retrain_total_steps,
                    completed=True,
                )


if __name__ == "__main__":
    main()
