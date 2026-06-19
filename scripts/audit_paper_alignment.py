from pathlib import Path
import argparse
import json
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
from flax import traverse_util

from classification_network import ImageNetResNet, MnistConvNet, ResNet56
from classification_network.engine import _kernel_decay_mask
from data import imagenet_stream_info
from transformation_network import CIFARAugmentationNetwork
from utils import load_config, validate_config


def _status(condition: bool, ok: str = "verified", fail: str = "missing") -> str:
    return ok if condition else fail


def _check(
    checks: List[Dict[str, Any]],
    check_id: str,
    requirement: str,
    status: str,
    evidence: List[str],
    details: Dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "id": check_id,
            "requirement": requirement,
            "status": status,
            "evidence": evidence,
            "details": details or {},
        }
    )


def _shape(value) -> List[int]:
    return [int(dim) for dim in value.shape]


def build_report(config_path: str) -> Dict[str, Any]:
    cfg = load_config(config_path)
    validate_config(cfg)

    checks: List[Dict[str, Any]] = []
    data_cfg = cfg["data"]
    classifier_cfg = cfg["classifier"]
    aug_cfg = cfg["augnet"]
    retrain_cfg = cfg["retrain"]

    augnet = CIFARAugmentationNetwork(
        tau_dim=aug_cfg["tau_dim"],
        tau_dropout=aug_cfg["tau_dropout"],
    )
    dummy_images = jnp.ones((2, 32, 32, 3), dtype=jnp.float32) * 0.5
    aug_variables = augnet.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        dummy_images,
        train=True,
        return_aux=True,
    )
    augmented, aux = augnet.apply(
        aug_variables,
        dummy_images,
        train=True,
        return_aux=True,
        rngs={"dropout": jax.random.PRNGKey(2)},
    )

    fields = aux["fields"]
    appearance_present = "appearance_delta" in aux
    _check(
        checks,
        "augnet_tau",
        "E predicts a 128-dimensional tau and applies 0.5 dropout during AugNet learning/retraining.",
        _status(aug_cfg["tau_dim"] == 128 and aug_cfg["tau_dropout"] == 0.5),
        [
            "configs/*_paper.yaml: augnet.tau_dim, augnet.tau_dropout",
            "transformation_network/models.py: AugmentationNetwork.__call__",
        ],
        {
            "tau_dim": aug_cfg["tau_dim"],
            "tau_dropout": aug_cfg["tau_dropout"],
            "runtime_tau_shape": _shape(aux["tau"]),
        },
    )
    _check(
        checks,
        "combined_spatial_appearance_transform",
        "G jointly emits spatial parameters sw/sb and RGB appearance parameters cw/cb.",
        _status(fields.shape[-1] == 18 and appearance_present),
        [
            "transformation_network/models.py: TransformationDecoder",
            "transformation_network/transforms.py: apply_spatial_transform/apply_appearance_transform",
        ],
        {
            "expected_channels": 18,
            "runtime_fields_shape": _shape(fields),
            "spatial_channels": 6,
            "appearance_channels": 12,
            "augmented_shape": _shape(augmented),
            "sample_grid_shape": _shape(aux["sample_grid"]),
            "appearance_delta_present": appearance_present,
        },
    )
    transforms_text = Path("transformation_network/transforms.py").read_text(encoding="utf-8")
    _check(
        checks,
        "field_smoothing",
        "Spatial and appearance transformation fields are smoothed with 4x4 average pooling before warping/filtering.",
        _status(
            augnet.smoothing_kernel == 4
            and "average_pool_same" in transforms_text
            and "smoothing_kernel" in transforms_text
        ),
        [
            "transformation_network/models.py: AugmentationNetwork.smoothing_kernel",
            "transformation_network/transforms.py: average_pool_same",
        ],
        {
            "smoothing_kernel": augnet.smoothing_kernel,
            "runtime_fields_shape": _shape(fields),
        },
    )

    classifier = ResNet56(
        num_classes=classifier_cfg["num_classes"],
        width_multiplier=classifier_cfg["width_multiplier"],
    )
    classifier_variables = classifier.init(
        jax.random.PRNGKey(3),
        jnp.ones((1, 32, 32, 3), dtype=jnp.float32),
        train=True,
    )
    classifier_params = classifier_variables["params"].get("classifier", {})
    _check(
        checks,
        "classifier_backbone",
        "CIFAR paper config uses ResNet-56 with a named final fully connected classifier layer.",
        _status(classifier_cfg["backbone"] == "resnet56" and "kernel" in classifier_params),
        [
            "configs/*_paper.yaml: classifier.backbone",
            "classification_network/models/backbones/resnet_cifar.py",
            "classification_network/models/heads/linear_head.py",
        ],
        {
            "backbone": classifier_cfg["backbone"],
            "num_classes": classifier_cfg["num_classes"],
            "classifier_kernel_shape": _shape(classifier_params["kernel"]) if "kernel" in classifier_params else None,
        },
    )
    decay_mask = _kernel_decay_mask(classifier_variables["params"])
    flat_decay_mask = traverse_util.flatten_dict(decay_mask, sep="/")
    kernel_masks = {path: value for path, value in flat_decay_mask.items() if path.endswith("/kernel")}
    bias_or_scale_masks = {
        path: value
        for path, value in flat_decay_mask.items()
        if path.endswith("/bias") or path.endswith("/scale")
    }
    _check(
        checks,
        "kernel_only_weight_decay",
        "Classifier weight decay is applied only to convolution/dense kernels, not BatchNorm scale/bias or classifier bias.",
        _status(
            bool(kernel_masks)
            and all(bool(value) for value in kernel_masks.values())
            and not any(bool(value) for value in bias_or_scale_masks.values())
        ),
        ["classification_network/engine.py: _kernel_decay_mask/_make_optimizer"],
        {
            "decayed_kernel_leaves": len(kernel_masks),
            "non_decayed_bias_or_scale_leaves": len(bias_or_scale_masks),
            "classifier_kernel_decayed": bool(flat_decay_mask.get("classifier/kernel", False)),
            "classifier_bias_decayed": bool(flat_decay_mask.get("classifier/bias", False)),
        },
    )
    _check(
        checks,
        "last_layer_influence",
        "Influence is approximated through the final FC layer and a fixed iHVP vector s_test, with dense-solve regression coverage.",
        _status(aug_cfg["precompute_s_test"] and aug_cfg["cg_iters"] > 0 and aug_cfg["s_test_batches"] > 0),
        [
            "paramyield_network/influence.py",
            "paramyield_network/engine.py: compute_batch_s_test/augnet_influence_train_step",
            "scripts/train_cifar10.py: precompute_s_test residual logging",
            "scripts/train_imagenet_stream.py: precompute_s_test residual logging",
            "scripts/run_checks.py: s_test_matches_dense_solve",
        ],
        {
            "precompute_s_test": aug_cfg["precompute_s_test"],
            "cg_iters": aug_cfg["cg_iters"],
            "s_test_batches": aug_cfg["s_test_batches"],
            "damping": aug_cfg["damping"],
        },
    )
    _check(
        checks,
        "replacement_influence_objective",
        "AugNet optimizes I_aug = Iup(augmented) - Iup(original), logging estimated validation loss reduction as -I_aug.",
        "implemented",
        ["paramyield_network/engine.py: augnet_influence_train_step"],
        {
            "metric": "estimated_val_loss_reduction",
            "regularizer": "identity_l2_weight",
            "identity_l2_weight": aug_cfg["identity_l2_weight"],
        },
    )
    _check(
        checks,
        "ragan_pretraining",
        "G is pretrained against baseline augmented images with image and feature discriminators using RaGAN-style losses.",
        "implemented",
        ["transformation_network/engine.py: augnet_pretrain_step", "transformation_network/discriminators.py"],
        {
            "pretrain_steps": cfg["pretrain"]["steps"],
            "image_loss_weight": cfg["pretrain"]["image_loss_weight"],
            "feature_loss_weight": cfg["pretrain"]["feature_loss_weight"],
        },
    )
    _check(
        checks,
        "baseline_cifar_augmentation",
        "CIFAR retraining applies baseline crop/flip and Cutout before learned AugNet.",
        _status(retrain_cfg["cutout_size"] == 16 and data_cfg["normalize"]),
        ["classification_network/engine.py: classifier_train_step_with_augnet", "configs/*_paper.yaml: retrain.cutout_size"],
        {
            "retrain_cutout_size": retrain_cfg["cutout_size"],
            "normalize": data_cfg["normalize"],
        },
    )
    _check(
        checks,
        "cifar_data_protocol",
        "CIFAR configs define a train/hyperval split and direct official archive loading.",
        _status(data_cfg["name"] in ("cifar10", "cifar100") and data_cfg["source"] == "direct"),
        ["data/cifar.py", config_path],
        {
            "dataset": data_cfg["name"],
            "source": data_cfg["source"],
            "hyperval_size": data_cfg["hyperval_size"],
        },
    )

    table1_specs = {
        "configs/cifar10_table1_labels10.yaml": ("cifar10", 10, 10),
        "configs/cifar10_table1_labels100.yaml": ("cifar10", 10, 100),
        "configs/cifar100_table1_labels100.yaml": ("cifar100", 100, 100),
    }
    table1_details = []
    table1_ok = True
    for path, (dataset_name, classes, labels_per_class) in table1_specs.items():
        detail: Dict[str, Any] = {"path": path}
        try:
            table_cfg = load_config(path)
            validate_config(table_cfg)
            actual_labels_per_class = table_cfg["data"].get("train_labels_per_class")
            expected_train_size = classes * labels_per_class
            batch_sizes = {
                "classifier": table_cfg["classifier"]["batch_size"],
                "pretrain": table_cfg["pretrain"]["batch_size"],
                "augnet": table_cfg["augnet"]["batch_size"],
            }
            expected_eval_batches = (10_000 + batch_sizes["classifier"] - 1) // batch_sizes["classifier"]
            classifier_steps_per_epoch = (
                expected_train_size + batch_sizes["classifier"] - 1
            ) // batch_sizes["classifier"]
            cutout_sizes = {
                "classifier": table_cfg["classifier"]["cutout_size"],
                "pretrain": table_cfg["pretrain"]["cutout_size"],
                "retrain": table_cfg["retrain"]["cutout_size"],
            }
            ok = (
                table_cfg["data"]["name"] == dataset_name
                and actual_labels_per_class == labels_per_class
                and table_cfg["classifier"]["backbone"] == "resnet56"
                and all(size <= expected_train_size for size in batch_sizes.values())
                and all(size == 0 for size in cutout_sizes.values())
                and table_cfg["eval_batches"] == expected_eval_batches
            )
            detail.update(
                {
                    "dataset": table_cfg["data"]["name"],
                    "train_labels_per_class": actual_labels_per_class,
                    "expected_train_size": expected_train_size,
                    "batch_sizes": batch_sizes,
                    "classifier_steps_per_epoch": classifier_steps_per_epoch,
                    "eval_batches": table_cfg["eval_batches"],
                    "expected_eval_batches": expected_eval_batches,
                    "cutout_sizes": cutout_sizes,
                    "status": "ok" if ok else "mismatch",
                }
            )
            table1_ok = table1_ok and ok
        except Exception as exc:  # noqa: BLE001
            detail.update({"status": "error", "error": str(exc)})
            table1_ok = False
        table1_details.append(detail)
    _check(
        checks,
        "table1_low_label_protocol",
        "Table 1 low-label CIFAR settings use balanced train_labels_per_class sampling with ResNet-56.",
        _status(table1_ok),
        ["configs/cifar10_table1_labels10.yaml", "configs/cifar10_table1_labels100.yaml", "configs/cifar100_table1_labels100.yaml"],
        {"configs": table1_details},
    )
    mnist_table1_specs = {
        "configs/mnist_table1_labels60.yaml": 60,
        "configs/mnist_table1_labels600.yaml": 600,
    }
    mnist_details = []
    mnist_ok = True
    for path, labels_per_class in mnist_table1_specs.items():
        detail: Dict[str, Any] = {"path": path}
        try:
            mnist_cfg = load_config(path)
            validate_config(mnist_cfg)
            ok = (
                mnist_cfg["data"]["name"] == "mnist"
                and mnist_cfg["data"].get("source") == "direct"
                and mnist_cfg["data"].get("train_labels_per_class") == labels_per_class
                and mnist_cfg["classifier"]["backbone"] == "mnist_cnn"
                and mnist_cfg["classifier"]["num_classes"] == 10
                and mnist_cfg["classifier"].get("baseline_augmentation") == "crop"
                and mnist_cfg["pretrain"].get("baseline_augmentation") == "crop"
                and mnist_cfg["retrain"].get("baseline_augmentation") == "crop"
                and not mnist_cfg["augnet"].get("use_appearance", True)
                and mnist_cfg["augnet"]["tau_dim"] == 128
                and mnist_cfg["augnet"]["tau_dropout"] == 0.5
                and mnist_cfg["eval_batches"]
                == (10_000 + mnist_cfg["classifier"]["batch_size"] - 1) // mnist_cfg["classifier"]["batch_size"]
            )
            detail.update(
                {
                    "dataset": mnist_cfg["data"]["name"],
                    "source": mnist_cfg["data"].get("source"),
                    "train_labels_per_class": mnist_cfg["data"].get("train_labels_per_class"),
                    "expected_train_size": labels_per_class * 10,
                    "eval_batches": mnist_cfg["eval_batches"],
                    "expected_eval_batches": (
                        10_000 + mnist_cfg["classifier"]["batch_size"] - 1
                    )
                    // mnist_cfg["classifier"]["batch_size"],
                    "backbone": mnist_cfg["classifier"]["backbone"],
                    "baseline_augmentation": {
                        "classifier": mnist_cfg["classifier"].get("baseline_augmentation", True),
                        "pretrain": mnist_cfg["pretrain"].get("baseline_augmentation", True),
                        "retrain": mnist_cfg["retrain"].get("baseline_augmentation", True),
                    },
                    "status": "ok" if ok else "mismatch",
                }
            )
            mnist_ok = mnist_ok and ok
        except Exception as exc:  # noqa: BLE001
            detail.update({"status": "error", "error": str(exc)})
            mnist_ok = False
        mnist_details.append(detail)

    mnist_model = MnistConvNet(num_classes=10)
    mnist_variables = mnist_model.init(
        jax.random.PRNGKey(5),
        jnp.ones((1, 28, 28, 1), dtype=jnp.float32),
        train=True,
    )
    mnist_classifier_params = mnist_variables["params"].get("classifier", {})
    _check(
        checks,
        "table1_mnist_protocol",
        "Table 1 MNIST settings use direct IDX loading, a 4-layer CNN, 1%/10% balanced labeled subsets, and spatial-only AugNet.",
        _status(mnist_ok and "kernel" in mnist_classifier_params),
        [
            "configs/mnist_table1_labels60.yaml",
            "configs/mnist_table1_labels600.yaml",
            "data/cifar.py: load_mnist_direct",
            "classification_network/models/backbones/mnist_cnn.py",
        ],
        {
            "configs": mnist_details,
            "classifier_kernel_shape": _shape(mnist_classifier_params["kernel"]) if "kernel" in mnist_classifier_params else None,
        },
    )
    figure4_script = Path("scripts/visualize_tau_interpolation.py")
    _check(
        checks,
        "figure4_tau_interpolation_visualization",
        "Figure 4-style tau traversal can export original, spatial field, spatial image, appearance delta, and final image rows.",
        _status(figure4_script.exists()),
        ["scripts/visualize_tau_interpolation.py", "transformation_network/models.py: tau_override"],
        {
            "script": str(figure4_script),
            "output_format": "PPM grid",
            "rows": ["original", "spatial_flow", "spatial_image", "appearance_delta", "final_augmented"],
        },
    )
    table2_wrn_config = Path("configs/cifar10_table2_wrn28_10.yaml")
    table2_wrn_ok = False
    table2_wrn_details: Dict[str, Any] = {"path": str(table2_wrn_config)}
    try:
        wrn_cfg = load_config(str(table2_wrn_config))
        validate_config(wrn_cfg)
        table2_wrn_ok = (
            wrn_cfg["data"]["name"] == "cifar10"
            and wrn_cfg["classifier"]["backbone"] == "wide_resnet28_10"
            and wrn_cfg["classifier"]["dropout_rate"] == 0.3
            and wrn_cfg["classifier"]["cutout_size"] == 16
            and wrn_cfg["retrain"]["cutout_size"] == 16
        )
        table2_wrn_details.update(
            {
                "dataset": wrn_cfg["data"]["name"],
                "backbone": wrn_cfg["classifier"]["backbone"],
                "dropout_rate": wrn_cfg["classifier"]["dropout_rate"],
                "classifier_cutout_size": wrn_cfg["classifier"]["cutout_size"],
                "retrain_cutout_size": wrn_cfg["retrain"]["cutout_size"],
                "status": "ok" if table2_wrn_ok else "mismatch",
            }
        )
    except Exception as exc:  # noqa: BLE001
        table2_wrn_details.update({"status": "error", "error": str(exc)})
    _check(
        checks,
        "table2_wrn28_10_protocol",
        "Table 2 Wide-ResNet-28-10 CIFAR-10 setting uses standard crop/flip/Cutout plus learned AugNet.",
        _status(table2_wrn_ok),
        ["configs/cifar10_table2_wrn28_10.yaml", "classification_network/models/backbones/wide_resnet.py"],
        table2_wrn_details,
    )
    table2_shake_config = Path("configs/cifar10_table2_shakeshake26_2x96d.yaml")
    table2_shake_ok = False
    table2_shake_details: Dict[str, Any] = {"path": str(table2_shake_config)}
    try:
        shake_cfg = load_config(str(table2_shake_config))
        validate_config(shake_cfg)
        table2_shake_ok = (
            shake_cfg["data"]["name"] == "cifar10"
            and shake_cfg["classifier"]["backbone"] == "shakeshake26_2x96d"
            and shake_cfg["classifier"]["cutout_size"] == 16
            and shake_cfg["retrain"]["cutout_size"] == 16
        )
        table2_shake_details.update(
            {
                "dataset": shake_cfg["data"]["name"],
                "backbone": shake_cfg["classifier"]["backbone"],
                "classifier_epochs": shake_cfg["classifier"]["epochs"],
                "classifier_cutout_size": shake_cfg["classifier"]["cutout_size"],
                "retrain_cutout_size": shake_cfg["retrain"]["cutout_size"],
                "status": "ok" if table2_shake_ok else "mismatch",
            }
        )
    except Exception as exc:  # noqa: BLE001
        table2_shake_details.update({"status": "error", "error": str(exc)})
    _check(
        checks,
        "table2_shakeshake_protocol",
        "Table 2 Shake-Shake CIFAR-10 setting uses 26 2x96d with standard crop/flip/Cutout plus learned AugNet.",
        _status(table2_shake_ok),
        ["configs/cifar10_table2_shakeshake26_2x96d.yaml", "classification_network/models/backbones/shake_shake.py"],
        table2_shake_details,
    )
    table2_pyramid_config = Path("configs/cifar10_table2_pyramidnet_shakedrop.yaml")
    table2_pyramid_ok = False
    table2_pyramid_details: Dict[str, Any] = {"path": str(table2_pyramid_config)}
    try:
        pyramid_cfg = load_config(str(table2_pyramid_config))
        validate_config(pyramid_cfg)
        table2_pyramid_ok = (
            pyramid_cfg["data"]["name"] == "cifar10"
            and pyramid_cfg["classifier"]["backbone"] == "pyramidnet272_shakedrop"
            and pyramid_cfg["classifier"]["cutout_size"] == 16
            and pyramid_cfg["retrain"]["cutout_size"] == 16
            and pyramid_cfg["classifier"]["final_keep_prob"] == 0.5
        )
        table2_pyramid_details.update(
            {
                "dataset": pyramid_cfg["data"]["name"],
                "backbone": pyramid_cfg["classifier"]["backbone"],
                "classifier_epochs": pyramid_cfg["classifier"]["epochs"],
                "classifier_cutout_size": pyramid_cfg["classifier"]["cutout_size"],
                "final_keep_prob": pyramid_cfg["classifier"]["final_keep_prob"],
                "retrain_cutout_size": pyramid_cfg["retrain"]["cutout_size"],
                "status": "ok" if table2_pyramid_ok else "mismatch",
            }
        )
    except Exception as exc:  # noqa: BLE001
        table2_pyramid_details.update({"status": "error", "error": str(exc)})
    _check(
        checks,
        "table2_pyramidnet_shakedrop_protocol",
        "Table 2 PyramidNet+ShakeDrop CIFAR-10 setting uses standard crop/flip/Cutout plus learned AugNet.",
        _status(table2_pyramid_ok),
        [
            "configs/cifar10_table2_pyramidnet_shakedrop.yaml",
            "classification_network/models/backbones/pyramidnet_shakedrop.py",
        ],
        table2_pyramid_details,
    )

    table2_cifar100_specs = {
        "configs/cifar100_table2_wrn28_10.yaml": {
            "backbone": "wide_resnet28_10",
            "dropout_rate": 0.3,
        },
        "configs/cifar100_table2_shakeshake26_2x96d.yaml": {
            "backbone": "shakeshake26_2x96d",
        },
        "configs/cifar100_table2_pyramidnet_shakedrop.yaml": {
            "backbone": "pyramidnet272_shakedrop",
            "final_keep_prob": 0.5,
        },
    }
    table2_cifar100_ok = True
    table2_cifar100_details = []
    for path, expected in table2_cifar100_specs.items():
        detail: Dict[str, Any] = {"path": path}
        try:
            cifar100_cfg = load_config(path)
            validate_config(cifar100_cfg)
            classifier_cfg = cifar100_cfg["classifier"]
            ok = (
                cifar100_cfg["data"]["name"] == "cifar100"
                and classifier_cfg["num_classes"] == 100
                and classifier_cfg["backbone"] == expected["backbone"]
                and classifier_cfg["cutout_size"] == 16
                and cifar100_cfg["retrain"]["cutout_size"] == 16
            )
            if "dropout_rate" in expected:
                ok = ok and classifier_cfg.get("dropout_rate") == expected["dropout_rate"]
            if "final_keep_prob" in expected:
                ok = ok and classifier_cfg.get("final_keep_prob") == expected["final_keep_prob"]
            detail.update(
                {
                    "dataset": cifar100_cfg["data"]["name"],
                    "backbone": classifier_cfg["backbone"],
                    "classifier_epochs": classifier_cfg["epochs"],
                    "classifier_cutout_size": classifier_cfg["cutout_size"],
                    "retrain_cutout_size": cifar100_cfg["retrain"]["cutout_size"],
                    "dropout_rate": classifier_cfg.get("dropout_rate"),
                    "final_keep_prob": classifier_cfg.get("final_keep_prob"),
                    "status": "ok" if ok else "mismatch",
                }
            )
            table2_cifar100_ok = table2_cifar100_ok and ok
        except Exception as exc:  # noqa: BLE001
            detail.update({"status": "error", "error": str(exc)})
            table2_cifar100_ok = False
        table2_cifar100_details.append(detail)
    _check(
        checks,
        "table2_cifar100_protocol",
        "Table 2 CIFAR-100 settings cover WRN-28-10, Shake-Shake 26 2x96d, and PyramidNet+ShakeDrop with Cutout plus learned AugNet.",
        _status(table2_cifar100_ok),
        list(table2_cifar100_specs.keys()),
        {"configs": table2_cifar100_details},
    )

    imagenet_specs = {
        "configs/imagenet_resnet50_paper.yaml": "resnet50",
        "configs/imagenet_resnet200_paper.yaml": "resnet200",
    }
    imagenet_details = []
    imagenet_ok = True
    for path, backbone in imagenet_specs.items():
        detail: Dict[str, Any] = {"path": path}
        try:
            imagenet_cfg = load_config(path)
            validate_config(imagenet_cfg)
            classifier_cfg = imagenet_cfg["classifier"]
            augnet_cfg = imagenet_cfg["augnet"]
            data_cfg = imagenet_cfg["data"]
            ok = (
                data_cfg["name"] == "imagenet"
                and data_cfg["source"] == "tfds"
                and data_cfg["hyperval_size"] == 50000
                and data_cfg.get("baseline_preprocessing") == "inception"
                and classifier_cfg["backbone"] == backbone
                and classifier_cfg["batch_size"] == 4096
                and classifier_cfg["learning_rate"] == 1.6
                and classifier_cfg["lr_decay_epochs"] == [90, 180, 240]
                and augnet_cfg["tau_dim"] == 128
                and augnet_cfg["tau_dropout"] == 0.5
                and len(augnet_cfg.get("encoder_widths", [])) == 8
                and len(augnet_cfg.get("decoder_widths", [])) == 8
                and imagenet_cfg["pretrain"].get("progressive_image_sizes") == [32, 64, 128, 224]
                and imagenet_cfg["pretrain"].get("progressive_boundaries") == [2500, 5000, 7500]
            )
            detail.update(
                {
                    "dataset": data_cfg["name"],
                    "hyperval_size": data_cfg["hyperval_size"],
                    "baseline_preprocessing": data_cfg.get("baseline_preprocessing"),
                    "backbone": classifier_cfg["backbone"],
                    "batch_size": classifier_cfg["batch_size"],
                    "learning_rate": classifier_cfg["learning_rate"],
                    "lr_decay_epochs": classifier_cfg["lr_decay_epochs"],
                    "encoder_layers": len(augnet_cfg.get("encoder_widths", [])),
                "decoder_layers": len(augnet_cfg.get("decoder_widths", [])),
                "progressive_image_sizes": imagenet_cfg["pretrain"].get("progressive_image_sizes"),
                "progressive_boundaries": imagenet_cfg["pretrain"].get("progressive_boundaries"),
                "classifier_drop_last": classifier_cfg.get("drop_last", True),
                "retrain_drop_last": imagenet_cfg["retrain"].get("drop_last", True),
                "status": "ok" if ok else "mismatch",
            }
            )
            imagenet_ok = imagenet_ok and ok
        except Exception as exc:  # noqa: BLE001
            detail.update({"status": "error", "error": str(exc)})
            imagenet_ok = False
        imagenet_details.append(detail)

    imagenet_model = ImageNetResNet(
        stage_sizes=(1, 1, 1, 1),
        widths=(8, 16, 32, 64),
        stem_width=8,
        num_classes=1000,
    )
    imagenet_variables = imagenet_model.init(
        jax.random.PRNGKey(6),
        jnp.ones((1, 64, 64, 3), dtype=jnp.float32),
        train=True,
    )
    imagenet_classifier_params = imagenet_variables["params"].get("classifier", {})
    stream_debug_cfg = load_config("configs/imagenet_stream_debug.yaml")
    validate_config(stream_debug_cfg)
    stream_info = imagenet_stream_info(stream_debug_cfg["data"])
    stream_runner = Path("scripts/train_imagenet_stream.py")
    stream_data_module = Path("data/imagenet.py")
    stream_runner_text = stream_runner.read_text(encoding="utf-8") if stream_runner.exists() else ""
    stream_data_text = stream_data_module.read_text(encoding="utf-8") if stream_data_module.exists() else ""
    _check(
        checks,
        "imagenet_backbone_config_protocol",
        "ImageNet scaffold defines ResNet-50/200 configs, streaming TFDS preprocessing/resume, 50k hyper-validation, batch/lr decay, Top-5 metrics, and 8-layer E/G.",
        _status(
            imagenet_ok
            and "kernel" in imagenet_classifier_params
            and stream_runner.exists()
            and stream_data_module.exists()
            and stream_info.input_shape == (1, 64, 64, 3)
            and "--resume" in stream_runner_text
            and "skip_batches" in stream_data_text
        ),
        [
            "configs/imagenet_resnet50_paper.yaml",
            "configs/imagenet_resnet200_paper.yaml",
            "configs/imagenet_stream_debug.yaml",
            "data/imagenet.py",
            "scripts/train_imagenet_stream.py",
            "classification_network/models/backbones/imagenet_resnet.py",
            "transformation_network/engine.py: augnet_pretrain_step.progressive_image_size",
            "classification_network/engine.py: top_k_accuracy",
            "scripts/train_imagenet_stream.py: --resume",
            "data/imagenet.py: skip_batches",
        ],
        {
            "configs": imagenet_details,
            "debug_classifier_kernel_shape": (
                _shape(imagenet_classifier_params["kernel"])
                if "kernel" in imagenet_classifier_params
                else None
            ),
            "stream_debug": {
                "input_shape": list(stream_info.input_shape),
                "train_split": stream_info.train_split,
                "hyperval_split": stream_info.hyperval_split,
                "validation_split": stream_info.validation_split,
            },
            "stream_resume": {
                "runner_has_resume": "--resume" in stream_runner_text,
                "iterator_has_skip_batches": "skip_batches" in stream_data_text,
            },
            "runtime_gaps": [
                "full ResNet-50/200 validation metrics",
            ],
        },
    )

    paper_targets = load_config("configs/paper_targets.yaml")
    target_details = []
    targets_ok = True
    supported_metrics = {
        "baseline_error_percent",
        "augnet_error_percent",
        "baseline_top1_percent",
        "augnet_top1_percent",
        "baseline_top5_percent",
        "augnet_top5_percent",
    }
    for target in paper_targets.get("targets", []):
        detail: Dict[str, Any] = {
            "id": target.get("id"),
            "config": target.get("config"),
            "metric": target.get("metric"),
            "target": target.get("target"),
        }
        try:
            target_cfg_path = Path(target["config"])
            target_cfg = load_config(str(target_cfg_path))
            validate_config(target_cfg)
            ok = (
                target_cfg_path.exists()
                and target.get("metric") in supported_metrics
                and target.get("direction") in ("max", "min")
                and isinstance(target.get("target"), (int, float))
            )
            detail["status"] = "ok" if ok else "mismatch"
            targets_ok = targets_ok and ok
        except Exception as exc:  # noqa: BLE001
            detail.update({"status": "error", "error": str(exc)})
            targets_ok = False
        target_details.append(detail)
    _check(
        checks,
        "paper_target_manifest",
        "Paper target manifest tracks Table 1/2 error rates and Table 3 ImageNet Top-1/Top-5 targets for strict single-run or suite-level comparison after runs.",
        _status(targets_ok and len(target_details) == 15),
        [
            "configs/paper_targets.yaml",
            "scripts/compare_to_paper.py",
            "scripts/run_paper_suite.py",
            "scripts/preflight.py",
            "scripts/collect_results.py",
        ],
        {
            "target_count": len(target_details),
            "configs": target_details,
        },
    )
    _check(
        checks,
        "reported_full_benchmark_families",
        "Track remaining paper benchmark families beyond CIFAR ResNet-56 and Wide-ResNet-28-10.",
        "partial",
        ["README.md: Implementation Notes"],
        {
            "covered_now": [
                "MNIST 4-layer CNN low-label path",
                "CIFAR ResNet-56 path",
                "CIFAR Wide-ResNet-28-10 path",
                "CIFAR Shake-Shake 26 2x96d path",
                "CIFAR PyramidNet+ShakeDrop path",
                "CIFAR-100 Table 2 WRN/Shake-Shake/PyramidNet configs",
                "CIFAR-sized AugNet/influence training",
                "4x4 smoothing of learned spatial and appearance transformation fields",
                "ImageNet ResNet-50/200 config and backbone scaffold",
                "ImageNet streaming TFDS runner and Inception-style preprocessing",
                "ImageNet progressive AugNet pretraining schedule",
                "5-seed suite materialization and mean/std comparison path",
                "preflight checks for config scale, CIFAR checksums, and TFDS dependencies",
            ],
            "missing": [
                "full paper metrics",
            ],
        },
    )

    summary: Dict[str, int] = {}
    for item in checks:
        summary[item["status"]] = summary.get(item["status"], 0) + 1

    return {
        "config": config_path,
        "summary": summary,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_paper.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = build_report(args.config)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
