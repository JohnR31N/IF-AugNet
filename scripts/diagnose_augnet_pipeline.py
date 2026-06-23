from pathlib import Path
import argparse
import json
import sys
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import optax

from classification_network.engine import (
    _apply_baseline_augmentation,
    accuracy,
    normalize_images,
    top_k_accuracy,
)
from data import NumpyBatchIterator, load_dataset
from paramyield_network import compute_batch_s_test, compute_batch_s_test_residual
from paramyield_network.influence import influence_up_loss
from scripts.train_cifar10 import (
    build_augnet,
    build_classifier,
    build_normalization,
    create_augnet_state_from_config,
    create_classifier_state,
    evaluate,
    infer_input_shape,
)
from utils import load_config, restore_state, validate_config
from utils.metrics import latest_by_stage, load_jsonl_metrics


def _to_float(value) -> float:
    return float(jax.device_get(value))


def _mean_dict(rows: list[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def _print_section(title: str) -> None:
    print(f"\n[{title}]")


def _print_metrics(metrics: Dict[str, float]) -> None:
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6f}")


def _latest_logged_metrics(checkpoint_dir: Path) -> None:
    metrics_path = checkpoint_dir / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"metrics.jsonl: missing at {metrics_path}")
        return
    latest = latest_by_stage(load_jsonl_metrics(str(metrics_path)))
    for stage in (
        "classifier_eval",
        "pretrain_augnet_last",
        "precompute_s_test",
        "augnet_last",
        "retrained_classifier_eval",
    ):
        record = latest.get(stage)
        if record is None:
            continue
        compact = " ".join(
            f"{key}={float(value):.4f}"
            for key, value in sorted(record.get("metrics", {}).items())
            if isinstance(value, (int, float))
        )
        print(f"{stage}: {compact}")


def _classifier_metrics(classifier, classifier_state, images, labels, image_mean, image_std):
    features, logits = classifier.apply(
        {"params": classifier_state.params, "batch_stats": classifier_state.batch_stats},
        normalize_images(images, image_mean, image_std),
        train=False,
        return_features=True,
    )
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    return features, logits, {
        "loss": _to_float(loss),
        "accuracy": _to_float(accuracy(logits, labels)),
        "top5_accuracy": _to_float(top_k_accuracy(logits, labels)),
    }


def _image_delta_metrics(name: str, augmented, reference) -> Dict[str, float]:
    delta = augmented - reference
    return {
        f"{name}_identity_l2": _to_float(jnp.mean(jnp.square(delta))),
        f"{name}_abs_delta": _to_float(jnp.mean(jnp.abs(delta))),
        f"{name}_max_abs_delta": _to_float(jnp.max(jnp.abs(delta))),
        f"{name}_min_pixel": _to_float(jnp.min(augmented)),
        f"{name}_max_pixel": _to_float(jnp.max(augmented)),
    }


def _batch_diagnostics(
    cfg,
    classifier,
    classifier_state,
    augnet,
    aug_state,
    train_batch,
    val_batch,
    rng,
    image_mean,
    image_std,
) -> Dict[str, float]:
    labels = train_batch["label"].astype(jnp.int32)
    images = train_batch["image"]
    classifier_cfg = cfg["classifier"]
    retrain_cfg = cfg["retrain"]
    aug_cfg = cfg["augnet"]

    rng_baseline, rng_raw, rng_base = jax.random.split(rng, 3)
    baseline_images = _apply_baseline_augmentation(
        images,
        rng_baseline,
        retrain_cfg.get("baseline_augmentation", classifier_cfg.get("baseline_augmentation", True)),
        retrain_cfg.get("cutout_size", classifier_cfg.get("cutout_size", 0)),
    )
    raw_aug, raw_aux = augnet.apply(
        {"params": aug_state.params},
        images,
        train=True,
        return_aux=True,
        rngs={"dropout": rng_raw},
    )
    baseline_aug, baseline_aux = augnet.apply(
        {"params": aug_state.params},
        baseline_images,
        train=True,
        return_aux=True,
        rngs={"dropout": rng_base},
    )

    original_features, _, original_metrics = _classifier_metrics(
        classifier,
        classifier_state,
        images,
        labels,
        image_mean,
        image_std,
    )
    baseline_features, _, baseline_metrics = _classifier_metrics(
        classifier,
        classifier_state,
        baseline_images,
        labels,
        image_mean,
        image_std,
    )
    raw_aug_features, _, raw_aug_metrics = _classifier_metrics(
        classifier,
        classifier_state,
        raw_aug,
        labels,
        image_mean,
        image_std,
    )
    baseline_aug_features, _, baseline_aug_metrics = _classifier_metrics(
        classifier,
        classifier_state,
        baseline_aug,
        labels,
        image_mean,
        image_std,
    )

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
    residual = compute_batch_s_test_residual(
        classifier_state,
        classifier,
        train_batch,
        val_batch,
        s_test,
        damping=aug_cfg["damping"],
        image_mean=image_mean,
        image_std=image_std,
    )
    original_iup = jnp.mean(
        influence_up_loss(
            original_features,
            labels,
            classifier_state.params["classifier"],
            s_test,
        )
    )
    raw_iup = jnp.mean(
        influence_up_loss(
            raw_aug_features,
            labels,
            classifier_state.params["classifier"],
            s_test,
        )
    )
    baseline_iup = jnp.mean(
        influence_up_loss(
            baseline_aug_features,
            labels,
            classifier_state.params["classifier"],
            s_test,
        )
    )

    metrics = {
        "original_loss": original_metrics["loss"],
        "original_accuracy": original_metrics["accuracy"],
        "baseline_loss": baseline_metrics["loss"],
        "baseline_accuracy": baseline_metrics["accuracy"],
        "raw_aug_loss": raw_aug_metrics["loss"],
        "raw_aug_accuracy": raw_aug_metrics["accuracy"],
        "baseline_aug_loss": baseline_aug_metrics["loss"],
        "baseline_aug_accuracy": baseline_aug_metrics["accuracy"],
        "s_test_residual": _to_float(residual),
        "original_iup": _to_float(original_iup),
        "raw_aug_iup": _to_float(raw_iup),
        "baseline_aug_iup": _to_float(baseline_iup),
        "raw_aug_delta_iup": _to_float(raw_iup - original_iup),
        "baseline_aug_delta_iup": _to_float(baseline_iup - original_iup),
        "raw_aug_estimated_reduction": _to_float(-(raw_iup - original_iup)),
        "baseline_aug_estimated_reduction": _to_float(-(baseline_iup - original_iup)),
        "raw_tau_abs_mean": _to_float(jnp.mean(jnp.abs(raw_aux["tau"]))),
        "baseline_tau_abs_mean": _to_float(jnp.mean(jnp.abs(baseline_aux["tau"]))),
    }
    metrics.update(_image_delta_metrics("baseline", baseline_images, images))
    metrics.update(_image_delta_metrics("raw_aug", raw_aug, images))
    metrics.update(_image_delta_metrics("baseline_aug", baseline_aug, baseline_images))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    classifier_ckpt = checkpoint_dir / "classifier.msgpack"
    augnet_ckpt = checkpoint_dir / "augnet.msgpack"
    retrained_ckpt = checkpoint_dir / "classifier_retrained.msgpack"

    _print_section("logged stage metrics")
    _latest_logged_metrics(checkpoint_dir)

    data_cfg = cfg["data"]
    splits = load_dataset(data_cfg, seed=cfg["seed"])
    input_shape = infer_input_shape(splits)
    image_mean, image_std = build_normalization(data_cfg)
    print(f"train_examples: {len(splits.train_images)}")
    print(f"hyperval_examples: {len(splits.hyperval_images)}")
    print(f"test_examples: {len(splits.test_images)}")

    classifier = build_classifier(cfg["classifier"])
    classifier_state = create_classifier_state(
        jax.random.PRNGKey(cfg["seed"]),
        classifier,
        input_shape=input_shape,
        learning_rate=cfg["classifier"]["learning_rate"],
        optimizer=cfg["classifier"]["optimizer"],
        momentum=cfg["classifier"]["momentum"],
        weight_decay=cfg["classifier"]["weight_decay"],
    )
    classifier_state = restore_state(str(classifier_ckpt), classifier_state, restore_opt_state=False)

    augnet = build_augnet(cfg, input_shape)
    aug_state = create_augnet_state_from_config(
        jax.random.PRNGKey(cfg["seed"] + 1),
        augnet,
        input_shape,
        cfg["augnet"],
    )
    aug_state = restore_state(str(augnet_ckpt), aug_state, restore_opt_state=False)

    if args.eval_batches > 0:
        _print_section("fresh classifier eval")
        test_iter = NumpyBatchIterator(
            splits.test_images,
            splits.test_labels,
            cfg["classifier"]["batch_size"],
            seed=cfg["seed"],
            shuffle=False,
            drop_last=False,
        )
        _print_metrics(
            evaluate(
                classifier_state,
                classifier,
                test_iter,
                args.eval_batches,
                image_mean=image_mean,
                image_std=image_std,
            )
        )

    train_iter = NumpyBatchIterator(
        splits.train_images,
        splits.train_labels,
        cfg["augnet"]["batch_size"],
        seed=cfg["seed"] + 11,
    )
    val_iter = NumpyBatchIterator(
        splits.hyperval_images,
        splits.hyperval_labels,
        cfg["augnet"]["batch_size"],
        seed=cfg["seed"] + 12,
    )
    rows = []
    for batch_index in range(args.batches):
        rows.append(
            _batch_diagnostics(
                cfg,
                classifier,
                classifier_state,
                augnet,
                aug_state,
                next(train_iter),
                next(val_iter),
                jax.random.PRNGKey(10_000 + batch_index),
                image_mean,
                image_std,
            )
        )

    _print_section("one-stage batch diagnostics")
    _print_metrics(_mean_dict(rows))

    if retrained_ckpt.exists() and args.eval_batches > 0:
        retrained_state = create_classifier_state(
            jax.random.PRNGKey(cfg["seed"] + 2),
            classifier,
            input_shape=input_shape,
            learning_rate=cfg["retrain"]["learning_rate"],
            optimizer=cfg["retrain"]["optimizer"],
            momentum=cfg["retrain"]["momentum"],
            weight_decay=cfg["retrain"]["weight_decay"],
        )
        retrained_state = restore_state(str(retrained_ckpt), retrained_state, restore_opt_state=False)
        _print_section("fresh retrained classifier eval")
        test_iter = NumpyBatchIterator(
            splits.test_images,
            splits.test_labels,
            cfg["classifier"]["batch_size"],
            seed=cfg["seed"],
            shuffle=False,
            drop_last=False,
        )
        _print_metrics(
            evaluate(
                retrained_state,
                classifier,
                test_iter,
                args.eval_batches,
                image_mean=image_mean,
                image_std=image_std,
            )
        )

    _print_section("readout")
    print(json.dumps(_mean_dict(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
