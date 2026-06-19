from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(config: Dict[str, Any]) -> None:
    dataset = config["data"]["name"]
    num_classes = config["classifier"]["num_classes"]
    expected_classes = {
        "cifar10": 10,
        "cifar100": 100,
        "imagenet": 1000,
        "mnist": 10,
        "synthetic_cifar": config["data"].get("num_classes", num_classes),
        "synthetic_imagenet": config["data"].get("num_classes", num_classes),
        "synthetic_mnist": config["data"].get("num_classes", num_classes),
    }.get(dataset)

    if expected_classes is None:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if num_classes != expected_classes:
        raise ValueError(
            f"classifier.num_classes={num_classes} does not match {dataset} classes={expected_classes}."
        )

    for section in ("classifier", "pretrain", "augnet"):
        batch_size = config[section]["batch_size"]
        if batch_size <= 0:
            raise ValueError(f"{section}.batch_size must be positive.")

    if config["data"]["hyperval_size"] <= 0:
        raise ValueError("data.hyperval_size must be positive.")

    learned_aug_input = config.get("retrain", {}).get("learned_aug_input", "baseline")
    if learned_aug_input not in ("raw", "baseline"):
        raise ValueError("retrain.learned_aug_input must be either 'raw' or 'baseline'.")

    train_labels_per_class = config["data"].get("train_labels_per_class")
    if train_labels_per_class is not None:
        if train_labels_per_class <= 0:
            raise ValueError("data.train_labels_per_class must be positive.")
        if config["data"].get("max_train_size") is not None:
            raise ValueError("data.train_labels_per_class cannot be combined with data.max_train_size.")

        estimated_train_size = train_labels_per_class * expected_classes
        for section in ("classifier", "pretrain", "augnet"):
            batch_size = config[section]["batch_size"]
            if batch_size > estimated_train_size:
                raise ValueError(
                    f"{section}.batch_size={batch_size} exceeds low-label training size "
                    f"{estimated_train_size} from data.train_labels_per_class."
                )
