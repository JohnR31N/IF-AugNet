from pathlib import Path
import argparse
import csv
import json
import sys
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.imagenet import IMAGENET_TRAIN_EXAMPLES, IMAGENET_VALIDATION_EXAMPLES
from utils import (
    latest_by_stage,
    load_config,
    load_jsonl_metrics,
    load_pickle,
    output_dir_from_arg,
    validate_config,
)


DEFAULT_CONFIGS = [
    "configs/mnist_table1_labels60.yaml",
    "configs/mnist_table1_labels600.yaml",
    "configs/cifar10_table1_labels10.yaml",
    "configs/cifar10_table1_labels100.yaml",
    "configs/cifar100_table1_labels100.yaml",
]


def _stage_completed(run_dir: Path, stage: str) -> bool:
    progress = run_dir / f"{stage}_progress.pkl"
    if not progress.exists():
        return False
    try:
        return bool(load_pickle(str(progress)).get("completed", False))
    except Exception:  # noqa: BLE001
        return False


def _metric(record: Dict[str, Any] | None, name: str) -> float | None:
    if record is None:
        return None
    value = record.get("metrics", {}).get(name)
    return None if value is None else float(value)


def _error_percent(accuracy: float | None) -> float | None:
    if accuracy is None:
        return None
    return (1.0 - accuracy) * 100.0


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _estimate_train_examples(config: Dict[str, Any]) -> int | None:
    data_cfg = config["data"]
    classifier_cfg = config["classifier"]
    dataset = data_cfg["name"]
    labels_per_class = data_cfg.get("train_labels_per_class")
    if labels_per_class is not None:
        return int(labels_per_class) * int(classifier_cfg["num_classes"])
    if data_cfg.get("max_train_size") is not None:
        return int(data_cfg["max_train_size"])
    if dataset in ("synthetic_cifar", "synthetic_mnist", "synthetic_imagenet"):
        return int(data_cfg.get("train_size", 0))
    if dataset in ("cifar10", "cifar100"):
        return 50_000 - int(data_cfg["hyperval_size"])
    if dataset == "mnist":
        return 60_000 - int(data_cfg["hyperval_size"])
    if dataset == "imagenet":
        return int(data_cfg.get("train_examples", IMAGENET_TRAIN_EXAMPLES - int(data_cfg["hyperval_size"])))
    return None


def _estimate_test_examples(config: Dict[str, Any]) -> int | None:
    data_cfg = config["data"]
    dataset = data_cfg["name"]
    if data_cfg.get("max_test_size") is not None:
        return int(data_cfg["max_test_size"])
    if dataset in ("synthetic_cifar", "synthetic_mnist", "synthetic_imagenet"):
        return int(data_cfg.get("test_size", 0))
    if dataset in ("cifar10", "cifar100", "mnist"):
        return 10_000
    if dataset == "imagenet":
        return int(data_cfg.get("validation_examples", IMAGENET_VALIDATION_EXAMPLES))
    return None


def _drop_last_for_section(config: Dict[str, Any], section: str) -> bool:
    default_drop_last = config["data"]["name"] == "imagenet"
    return bool(config[section].get("drop_last", default_drop_last))


def _steps_per_epoch(config: Dict[str, Any], section: str, train_examples: int | None) -> int | None:
    if train_examples is None:
        return None
    batch_size = int(config[section].get("batch_size", config["classifier"]["batch_size"]))
    if _drop_last_for_section(config, section):
        return max(1, train_examples // batch_size)
    return max(1, (train_examples + batch_size - 1) // batch_size)


def _eval_batches(config: Dict[str, Any], test_examples: int | None) -> int | None:
    if test_examples is None:
        return None
    batch_size = int(config["classifier"]["batch_size"])
    return max(1, (test_examples + batch_size - 1) // batch_size)


def collect_config_result(config_path: str) -> Dict[str, Any]:
    cfg = load_config(config_path)
    validate_config(cfg)

    run_dir = Path(cfg["checkpoint_dir"])
    metrics_path = run_dir / "metrics.jsonl"
    data_cfg = cfg["data"]
    classifier_cfg = cfg["classifier"]
    labels_per_class = data_cfg.get("train_labels_per_class")
    train_examples = _estimate_train_examples(cfg)
    test_examples = _estimate_test_examples(cfg)
    classifier_steps_per_epoch = _steps_per_epoch(cfg, "classifier", train_examples)
    retrain_steps_per_epoch = _steps_per_epoch(cfg, "retrain", train_examples)
    expected_eval_batches = _eval_batches(cfg, test_examples)

    row: Dict[str, Any] = {
        "config": config_path,
        "run_dir": str(run_dir),
        "dataset": data_cfg["name"],
        "labels_per_class": labels_per_class,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "classifier_batch_size": classifier_cfg["batch_size"],
        "eval_batches": cfg.get("eval_batches"),
        "expected_eval_batches": expected_eval_batches,
        "classifier_steps_per_epoch": classifier_steps_per_epoch,
        "classifier_total_steps": (
            classifier_steps_per_epoch * int(classifier_cfg["epochs"])
            if classifier_steps_per_epoch is not None
            else None
        ),
        "retrain_steps_per_epoch": retrain_steps_per_epoch,
        "retrain_total_steps": (
            retrain_steps_per_epoch * int(cfg["retrain"]["epochs"])
            if retrain_steps_per_epoch is not None
            else None
        ),
        "classifier": classifier_cfg["backbone"],
        "status": "missing_metrics",
        "classifier_completed": _stage_completed(run_dir, "classifier"),
        "pretrain_augnet_completed": _stage_completed(run_dir, "pretrain_augnet"),
        "augnet_completed": _stage_completed(run_dir, "augnet"),
        "retrain_completed": _stage_completed(run_dir, "retrain"),
        "baseline_accuracy": None,
        "baseline_top5_accuracy": None,
        "baseline_error": None,
        "augnet_accuracy": None,
        "augnet_top5_accuracy": None,
        "augnet_error": None,
        "error_reduction": None,
        "baseline_loss": None,
        "augnet_loss": None,
        "s_test_batches": None,
        "s_test_cg_iters": None,
        "s_test_damping": None,
        "s_test_residual_mean": None,
        "s_test_residual_max": None,
        "pretrain_tau_abs_mean": None,
        "pretrain_identity_l2": None,
        "augnet_tau_abs_mean": None,
        "augnet_identity_l2": None,
        "estimated_val_loss_reduction": None,
    }

    if not metrics_path.exists():
        return row

    records = load_jsonl_metrics(str(metrics_path))
    latest = latest_by_stage(records)
    baseline_eval = latest.get("classifier_eval")
    augnet_eval = latest.get("retrained_classifier_eval")
    precompute_s_test = latest.get("precompute_s_test")
    pretrain_last = latest.get("pretrain_augnet_last")
    augnet_last = latest.get("augnet_last")

    baseline_accuracy = _metric(baseline_eval, "accuracy")
    baseline_top5_accuracy = _metric(baseline_eval, "top5_accuracy")
    augnet_accuracy = _metric(augnet_eval, "accuracy")
    augnet_top5_accuracy = _metric(augnet_eval, "top5_accuracy")
    baseline_error = _error_percent(baseline_accuracy)
    augnet_error = _error_percent(augnet_accuracy)

    row.update(
        {
            "status": "ok" if baseline_eval and augnet_eval else "missing_final_eval",
            "baseline_accuracy": baseline_accuracy,
            "baseline_top5_accuracy": baseline_top5_accuracy,
            "baseline_error": baseline_error,
            "augnet_accuracy": augnet_accuracy,
            "augnet_top5_accuracy": augnet_top5_accuracy,
            "augnet_error": augnet_error,
            "error_reduction": (
                baseline_error - augnet_error
                if baseline_error is not None and augnet_error is not None
                else None
            ),
            "baseline_loss": _metric(baseline_eval, "loss"),
            "augnet_loss": _metric(augnet_eval, "loss"),
            "s_test_batches": _metric(precompute_s_test, "batches"),
            "s_test_cg_iters": _metric(precompute_s_test, "cg_iters"),
            "s_test_damping": _metric(precompute_s_test, "damping"),
            "s_test_residual_mean": _metric(precompute_s_test, "residual_mean"),
            "s_test_residual_max": _metric(precompute_s_test, "residual_max"),
            "pretrain_tau_abs_mean": _metric(pretrain_last, "pretrain_tau_abs_mean"),
            "pretrain_identity_l2": _metric(pretrain_last, "pretrain_identity_l2"),
            "augnet_tau_abs_mean": _metric(augnet_last, "tau_abs_mean"),
            "augnet_identity_l2": _metric(augnet_last, "identity_l2"),
            "estimated_val_loss_reduction": _metric(augnet_last, "estimated_val_loss_reduction"),
        }
    )
    return row


def collect_results(configs: Iterable[str]) -> List[Dict[str, Any]]:
    return [collect_config_result(config_path) for config_path in configs]


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "run_dir",
        "dataset",
        "labels_per_class",
        "train_examples",
        "test_examples",
        "classifier_batch_size",
        "eval_batches",
        "expected_eval_batches",
        "classifier_steps_per_epoch",
        "classifier_total_steps",
        "retrain_steps_per_epoch",
        "retrain_total_steps",
        "classifier",
        "status",
        "baseline_error",
        "augnet_error",
        "error_reduction",
        "baseline_accuracy",
        "baseline_top5_accuracy",
        "augnet_accuracy",
        "augnet_top5_accuracy",
        "baseline_loss",
        "augnet_loss",
        "s_test_batches",
        "s_test_cg_iters",
        "s_test_damping",
        "s_test_residual_mean",
        "s_test_residual_max",
        "pretrain_tau_abs_mean",
        "pretrain_identity_l2",
        "augnet_tau_abs_mean",
        "augnet_identity_l2",
        "estimated_val_loss_reduction",
        "classifier_completed",
        "pretrain_augnet_completed",
        "augnet_completed",
        "retrain_completed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("dataset", "Dataset"),
        ("labels_per_class", "Labels/class"),
        ("train_examples", "Train examples"),
        ("test_examples", "Test examples"),
        ("eval_batches", "Eval batches"),
        ("classifier", "Classifier"),
        ("status", "Status"),
        ("baseline_error", "Baseline error"),
        ("augnet_error", "AugNet error"),
        ("error_reduction", "Error reduction"),
        ("baseline_top5_accuracy", "Baseline Top-5"),
        ("augnet_top5_accuracy", "AugNet Top-5"),
        ("s_test_residual_mean", "s_test residual"),
        ("estimated_val_loss_reduction", "Est. val reduction"),
    ]
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[key]) for key, _ in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", default="runs/results")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = collect_results(args.configs)
    try:
        output_dir = output_dir_from_arg(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    write_csv(rows, output_dir / "results_table.csv")
    write_markdown(rows, output_dir / "results_table.md")
    (output_dir / "results_table.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {output_dir / 'results_table.csv'}")
    print(f"wrote {output_dir / 'results_table.md'}")
    print(f"wrote {output_dir / 'results_table.json'}")

    if args.strict and any(row["status"] != "ok" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
