from pathlib import Path
import argparse
import sys
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import numpy as np
from tqdm import trange

from classification_network import (
    classifier_train_step,
    classifier_train_step_with_augnet,
    create_classifier_state,
)
from data import imagenet_stream_info, make_imagenet_iterator
from paramyield_network import (
    augnet_influence_train_step,
    compute_batch_s_test,
    compute_batch_s_test_residual,
)
from transformation_network import (
    FeatureDiscriminator,
    ImageDiscriminator,
    augnet_pretrain_step,
    create_discriminator_state,
)
from scripts.train_cifar10 import (
    _format_metrics,
    _to_float,
    build_augnet,
    build_classifier,
    build_learning_rate,
    build_normalization,
    create_augnet_state_from_config,
    evaluate,
    pretrain_augnet_optimizer_config,
    progressive_image_size_for_step,
    rng_from_list,
    should_checkpoint,
    stage_completed,
    stage_paths,
    stop_step_for_stage,
    steps_per_epoch,
    tree_average,
)
from utils import JsonlMetricLogger, load_config, load_pickle, restore_state, save_state, validate_config, write_run_manifest


def _rng_to_list(rng) -> list:
    return np.asarray(rng).tolist()


def _restore_stream_progress(checkpoint_dir: Path, stage: str, state, rng):
    paths = stage_paths(checkpoint_dir, stage)
    if not paths["progress"].exists():
        return state, rng, 0
    progress = load_pickle(str(paths["progress"]))
    if paths["state"].exists():
        state = restore_state(str(paths["state"]), state)
    restored_rng = rng_from_list(progress["rng"]) if "rng" in progress else rng
    return state, restored_rng, int(progress.get("next_step", 0))


def _infer_feature_dim(state, model, input_shape) -> int:
    features, _ = model.apply(
        {"params": state.params, "batch_stats": state.batch_stats},
        np.ones(input_shape, np.float32),
        train=False,
        return_features=True,
    )
    return int(features.shape[-1])


def _stream_drop_last(config: Dict, section: str) -> bool:
    return bool(config[section].get("drop_last", True))


def _save_latest(checkpoint_dir: Path, stage: str, state, rng, step: int, total_steps: int) -> None:
    save_state(str(checkpoint_dir / f"{stage}_latest.msgpack"), state)
    # Streaming TFDS iterators do not expose resumable sample positions; keep RNG/step metadata only.
    from utils import save_pickle

    save_pickle(
        str(checkpoint_dir / f"{stage}_progress.pkl"),
        {
            "completed": step >= total_steps,
            "next_step": step,
            "rng": _rng_to_list(rng),
            "total_steps": total_steps,
            "streaming_iterator": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", default="configs/imagenet_resnet50_paper.yaml")
    parser.add_argument(
        "--stage",
        choices=("classifier", "pretrain_augnet", "augnet", "retrain", "all"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
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
    if cfg["data"]["name"] != "imagenet":
        raise ValueError("train_imagenet_stream.py expects data.name: imagenet.")

    info = imagenet_stream_info(cfg["data"])
    input_shape = info.input_shape
    classifier_cfg = cfg["classifier"]
    if not _stream_drop_last(cfg, "classifier") or not _stream_drop_last(cfg, "retrain"):
        raise ValueError(
            "ImageNet streaming training currently requires drop_last=True for classifier "
            "and retrain stages to keep large-batch shapes stable."
        )
    classifier_steps_per_epoch = steps_per_epoch(
        info.train_examples,
        classifier_cfg["batch_size"],
        drop_last=_stream_drop_last(cfg, "classifier"),
    )
    retrain_batch_size = cfg["retrain"].get("batch_size", classifier_cfg["batch_size"])
    retrain_steps_per_epoch = steps_per_epoch(
        info.train_examples,
        retrain_batch_size,
        drop_last=_stream_drop_last(cfg, "retrain"),
    )

    if args.dry_run:
        print(
            "imagenet_stream_dry_run "
            f"input_shape={input_shape} "
            f"train_split={info.train_split} "
            f"hyperval_split={info.hyperval_split} "
            f"validation_split={info.validation_split} "
            f"classifier_steps_per_epoch={classifier_steps_per_epoch} "
            f"retrain_steps_per_epoch={retrain_steps_per_epoch} "
            f"eval_batches={cfg['eval_batches']}"
        )
        return

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every_steps = cfg.get("checkpoint_every_steps", 0)
    metric_logger = JsonlMetricLogger(str(checkpoint_dir / "metrics.jsonl"))
    write_run_manifest(str(checkpoint_dir / "run_manifest.json"), cfg, " ".join(sys.argv))

    image_mean, image_std = build_normalization(cfg["data"])

    rng = jax.random.PRNGKey(cfg["seed"])
    rng_model, rng_aug, rng_image_d, rng_feature_d, rng_retrain = jax.random.split(rng, 5)

    classifier = build_classifier(classifier_cfg)
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

    classifier_ckpt = checkpoint_dir / "classifier.msgpack"
    augnet_ckpt = checkpoint_dir / "augnet.msgpack"
    retrained_classifier_ckpt = checkpoint_dir / "classifier_retrained.msgpack"

    if args.stage in ("pretrain_augnet", "augnet"):
        classifier_state = restore_state(str(classifier_ckpt), classifier_state)
    if args.stage == "retrain" and augnet_ckpt.exists():
        aug_state = restore_state(str(augnet_ckpt), aug_state, restore_opt_state=False)

    if args.stage in ("classifier", "all"):
        total_steps = classifier_cfg["epochs"] * classifier_steps_per_epoch
        classifier_start_step = 0
        if args.resume and stage_completed(checkpoint_dir, "classifier"):
            classifier_state = restore_state(str(classifier_ckpt), classifier_state)
            metric_logger.log("resume_classifier", {"skipped": 1.0})
            classifier_start_step = total_steps
        elif args.resume:
            classifier_state, rng, classifier_start_step = _restore_stream_progress(
                checkpoint_dir,
                "classifier",
                classifier_state,
                rng,
            )
            if classifier_start_step:
                metric_logger.log(
                    "resume_classifier",
                    {"next_step": float(classifier_start_step)},
                )

        if classifier_start_step < total_steps:
            classifier_stop_step = stop_step_for_stage(
                classifier_start_step,
                total_steps,
                args.stop_after_steps,
            )
            train_iter = make_imagenet_iterator(
                cfg["data"],
                split=info.train_split,
                batch_size=classifier_cfg["batch_size"],
                training=True,
                seed=cfg["seed"],
                skip_batches=classifier_start_step,
            )
            progress = trange(classifier_start_step, classifier_stop_step, desc="classifier")
            for step in progress:
                epoch = step // classifier_steps_per_epoch
                rng, step_rng = jax.random.split(rng)
                classifier_state, metrics = classifier_train_step(
                    classifier_state,
                    classifier,
                    next(train_iter),
                    step_rng,
                    apply_baseline_augmentation=False,
                    cutout_size=0,
                    image_mean=image_mean,
                    image_std=image_std,
                )
                if step % cfg["log_every"] == 0:
                    metrics_float = _to_float(metrics)
                    progress.set_postfix(metrics_float)
                    metric_logger.log("classifier_train", metrics_float, step=step, epoch=epoch + 1)
                if should_checkpoint(step, checkpoint_every_steps):
                    _save_latest(checkpoint_dir, "classifier", classifier_state, rng, step + 1, total_steps)

            if classifier_stop_step < total_steps:
                _save_latest(checkpoint_dir, "classifier", classifier_state, rng, classifier_stop_step, total_steps)
                print(
                    f"classifier stage paused at "
                    f"{classifier_stop_step}/{total_steps} steps"
                )
            else:
                eval_iter = make_imagenet_iterator(
                    cfg["data"],
                    split=info.validation_split,
                    batch_size=classifier_cfg["batch_size"],
                    training=False,
                    seed=cfg["seed"] + 10,
                    repeat=False,
                    drop_remainder=False,
                )
                eval_metrics = evaluate(
                    classifier_state,
                    classifier,
                    eval_iter,
                    cfg["eval_batches"],
                    image_mean=image_mean,
                    image_std=image_std,
                )
                print(_format_metrics("classifier_eval", eval_metrics))
                metric_logger.log("classifier_eval", eval_metrics)
                save_state(str(classifier_ckpt), classifier_state)
                _save_latest(checkpoint_dir, "classifier", classifier_state, rng, total_steps, total_steps)

    if args.stage in ("pretrain_augnet", "all"):
        pretrain_cfg = cfg["pretrain"]
        aug_state = create_augnet_state_from_config(
            rng_aug,
            augnet,
            input_shape,
            pretrain_augnet_optimizer_config(aug_cfg, pretrain_cfg),
        )
        pretrain_start_step = 0
        pretrain_completed = args.resume and stage_completed(checkpoint_dir, "pretrain_augnet")
        if pretrain_completed:
            aug_state = restore_state(str(augnet_ckpt), aug_state, restore_opt_state=False)
            metric_logger.log("resume_pretrain_augnet", {"skipped": 1.0})
            pretrain_start_step = pretrain_cfg["steps"]
        elif args.resume:
            aug_state, rng, pretrain_start_step = _restore_stream_progress(
                checkpoint_dir,
                "pretrain_augnet",
                aug_state,
                rng,
            )
            if pretrain_start_step:
                metric_logger.log(
                    "resume_pretrain_augnet",
                    {"next_step": float(pretrain_start_step)},
                )

        image_discriminator = ImageDiscriminator()
        feature_discriminator = FeatureDiscriminator()
        feature_dim = _infer_feature_dim(classifier_state, classifier, input_shape)
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
        pretrain_image_d_latest = checkpoint_dir / "pretrain_augnet_image_discriminator_latest.msgpack"
        pretrain_feature_d_latest = checkpoint_dir / "pretrain_augnet_feature_discriminator_latest.msgpack"
        if args.resume and pretrain_start_step and not pretrain_completed:
            if pretrain_image_d_latest.exists():
                image_discriminator_state = restore_state(
                    str(pretrain_image_d_latest),
                    image_discriminator_state,
                )
            if pretrain_feature_d_latest.exists():
                feature_discriminator_state = restore_state(
                    str(pretrain_feature_d_latest),
                    feature_discriminator_state,
                )

        if pretrain_start_step < pretrain_cfg["steps"]:
            pretrain_stop_step = stop_step_for_stage(
                pretrain_start_step,
                pretrain_cfg["steps"],
                args.stop_after_steps,
            )
            pretrain_iter = make_imagenet_iterator(
                cfg["data"],
                split=info.train_split,
                batch_size=pretrain_cfg["batch_size"],
                training=True,
                seed=cfg["seed"] + 3,
                skip_batches=pretrain_start_step,
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
                    apply_baseline_augmentation=False,
                    cutout_size=0,
                    progressive_image_size=progressive_image_size_for_step(
                        pretrain_cfg,
                        step,
                        input_shape[1],
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
                    _save_latest(checkpoint_dir, "pretrain_augnet", aug_state, rng, step + 1, pretrain_cfg["steps"])
                    save_state(str(pretrain_image_d_latest), image_discriminator_state)
                    save_state(str(pretrain_feature_d_latest), feature_discriminator_state)

            if pretrain_stop_step < pretrain_cfg["steps"]:
                _save_latest(checkpoint_dir, "pretrain_augnet", aug_state, rng, pretrain_stop_step, pretrain_cfg["steps"])
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
                _save_latest(checkpoint_dir, "pretrain_augnet", aug_state, rng, pretrain_cfg["steps"], pretrain_cfg["steps"])
                save_state(str(pretrain_image_d_latest), image_discriminator_state)
                save_state(str(pretrain_feature_d_latest), feature_discriminator_state)

    if args.stage in ("augnet", "all"):
        aug_state = create_augnet_state_from_config(rng_aug, augnet, input_shape, aug_cfg)
        if augnet_ckpt.exists():
            aug_state = restore_state(str(augnet_ckpt), aug_state, restore_opt_state=False)
        augnet_start_step = 0
        if args.resume and stage_completed(checkpoint_dir, "augnet"):
            aug_state = restore_state(str(augnet_ckpt), aug_state, restore_opt_state=False)
            metric_logger.log("resume_augnet", {"skipped": 1.0})
            augnet_start_step = aug_cfg["steps"]
        elif args.resume:
            aug_state, rng, augnet_start_step = _restore_stream_progress(
                checkpoint_dir,
                "augnet",
                aug_state,
                rng,
            )
            if augnet_start_step:
                metric_logger.log(
                    "resume_augnet",
                    {"next_step": float(augnet_start_step)},
                )

        if augnet_start_step < aug_cfg["steps"]:
            aug_train_iter = make_imagenet_iterator(
                cfg["data"],
                split=info.train_split,
                batch_size=aug_cfg["batch_size"],
                training=True,
                seed=cfg["seed"] + 1,
                skip_batches=augnet_start_step,
            )
            hyperval_iter = make_imagenet_iterator(
                cfg["data"],
                split=info.hyperval_split,
                batch_size=aug_cfg["batch_size"],
                training=False,
                seed=cfg["seed"] + 2,
                skip_batches=augnet_start_step,
            )

            fixed_s_test = None
            if aug_cfg.get("precompute_s_test", True):
                s_tests = []
                s_test_residuals = []
                s_test_train_iter = make_imagenet_iterator(
                    cfg["data"],
                    split=info.train_split,
                    batch_size=aug_cfg["batch_size"],
                    training=True,
                    seed=cfg["seed"] + 3,
                )
                s_test_hyperval_iter = make_imagenet_iterator(
                    cfg["data"],
                    split=info.hyperval_split,
                    batch_size=aug_cfg["batch_size"],
                    training=False,
                    seed=cfg["seed"] + 4,
                )
                progress = trange(aug_cfg.get("s_test_batches", 1), desc="precompute_s_test")
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
                        "batches": float(aug_cfg.get("s_test_batches", 1)),
                        "damping": float(aug_cfg["damping"]),
                        "cg_iters": float(aug_cfg["cg_iters"]),
                        "residual_mean": float(np.mean(s_test_residuals)),
                        "residual_max": float(np.max(s_test_residuals)),
                    },
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
                s_test = fixed_s_test
                if s_test is None:
                    s_test = compute_batch_s_test(
                        classifier_state,
                        classifier,
                        train_batch,
                        next(hyperval_iter),
                        damping=aug_cfg["damping"],
                        cg_iters=aug_cfg["cg_iters"],
                        image_mean=image_mean,
                        image_std=image_std,
                    )
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
                    _save_latest(checkpoint_dir, "augnet", aug_state, rng, step + 1, aug_cfg["steps"])

            if augnet_stop_step < aug_cfg["steps"]:
                _save_latest(checkpoint_dir, "augnet", aug_state, rng, augnet_stop_step, aug_cfg["steps"])
                print(
                    f"augnet stage paused at "
                    f"{augnet_stop_step}/{aug_cfg['steps']} steps"
                )
            else:
                metrics_float = _to_float(metrics)
                print(_format_metrics("augnet_last", metrics_float))
                metric_logger.log("augnet_last", metrics_float)
                save_state(str(augnet_ckpt), aug_state)
                _save_latest(checkpoint_dir, "augnet", aug_state, rng, aug_cfg["steps"], aug_cfg["steps"])

    if args.stage in ("retrain", "all"):
        retrain_cfg = cfg["retrain"]
        retrain_batch_size = retrain_cfg.get("batch_size", classifier_cfg["batch_size"])
        retrain_steps_per_epoch = steps_per_epoch(
            info.train_examples,
            retrain_batch_size,
            drop_last=_stream_drop_last(cfg, "retrain"),
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
        total_steps = retrain_cfg["epochs"] * retrain_steps_per_epoch
        retrain_start_step = 0
        if args.resume and stage_completed(checkpoint_dir, "retrain"):
            retrained_state = restore_state(str(retrained_classifier_ckpt), retrained_state)
            metric_logger.log("resume_retrain", {"skipped": 1.0})
            retrain_start_step = total_steps
        elif args.resume:
            retrained_state, rng, retrain_start_step = _restore_stream_progress(
                checkpoint_dir,
                "retrain",
                retrained_state,
                rng,
            )
            if retrain_start_step:
                metric_logger.log(
                    "resume_retrain",
                    {"next_step": float(retrain_start_step)},
                )

        if retrain_start_step < total_steps:
            retrain_stop_step = stop_step_for_stage(
                retrain_start_step,
                total_steps,
                args.stop_after_steps,
            )
            retrain_iter = make_imagenet_iterator(
                cfg["data"],
                split=info.train_split,
                batch_size=retrain_batch_size,
                training=True,
                seed=cfg["seed"] + 5,
                skip_batches=retrain_start_step,
            )
            progress = trange(retrain_start_step, retrain_stop_step, desc="retrain")
            for step in progress:
                epoch = step // retrain_steps_per_epoch
                rng, step_rng = jax.random.split(rng)
                retrained_state, metrics = classifier_train_step_with_augnet(
                    retrained_state,
                    classifier,
                    aug_state,
                    augnet,
                    next(retrain_iter),
                    step_rng,
                    apply_baseline_augmentation=False,
                    cutout_size=0,
                    learned_aug_probability=retrain_cfg.get("learned_aug_probability", 1.0),
                    learned_aug_input=retrain_cfg.get("learned_aug_input", "raw"),
                    image_mean=image_mean,
                    image_std=image_std,
                )
                if step % cfg["log_every"] == 0:
                    metrics_float = _to_float(metrics)
                    progress.set_postfix(metrics_float)
                    metric_logger.log("retrain", metrics_float, step=step, epoch=epoch + 1)
                if should_checkpoint(step, checkpoint_every_steps):
                    _save_latest(checkpoint_dir, "retrain", retrained_state, rng, step + 1, total_steps)

            if retrain_stop_step < total_steps:
                _save_latest(checkpoint_dir, "retrain", retrained_state, rng, retrain_stop_step, total_steps)
                print(
                    f"retrain stage paused at "
                    f"{retrain_stop_step}/{total_steps} steps"
                )
            else:
                eval_iter = make_imagenet_iterator(
                    cfg["data"],
                    split=info.validation_split,
                    batch_size=classifier_cfg["batch_size"],
                    training=False,
                    seed=cfg["seed"] + 11,
                    repeat=False,
                    drop_remainder=False,
                )
                eval_metrics = evaluate(
                    retrained_state,
                    classifier,
                    eval_iter,
                    cfg["eval_batches"],
                    image_mean=image_mean,
                    image_std=image_std,
                )
                print(_format_metrics("retrained_classifier_eval", eval_metrics))
                metric_logger.log("retrained_classifier_eval", eval_metrics)
                save_state(str(retrained_classifier_ckpt), retrained_state)
                _save_latest(checkpoint_dir, "retrain", retrained_state, rng, total_steps, total_steps)


if __name__ == "__main__":
    main()
