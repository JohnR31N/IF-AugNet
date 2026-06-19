from pathlib import Path
import argparse
import importlib.util
import json
import sys
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.cifar import (
    _CIFAR10_MD5,
    _CIFAR10_URLS,
    _CIFAR100_MD5,
    _CIFAR100_URLS,
    _MNIST_MIRRORS,
    _MNIST_RESOURCES,
    file_md5,
    load_cifar10_direct,
    load_cifar100_direct,
    load_mnist_direct,
)
from data.imagenet import IMAGENET_TRAIN_EXAMPLES, IMAGENET_VALIDATION_EXAMPLES, imagenet_stream_info
from scripts.compare_to_paper import DEFAULT_TARGETS, filter_paper_targets, load_paper_targets
from scripts.run_paper_suite import unique_target_configs
from utils import load_config, validate_config


_STATUS_RANK = {
    "ok": 0,
    "warning": 1,
    "error": 2,
}


def _overall_status(checks: Iterable[Dict[str, Any]]) -> str:
    status = "ok"
    for check in checks:
        if _STATUS_RANK[check["status"]] > _STATUS_RANK[status]:
            status = check["status"]
    return status


def _check(checks: List[Dict[str, Any]], check_id: str, status: str, message: str, **details: Any) -> None:
    checks.append(
        {
            "id": check_id,
            "status": status,
            "message": message,
            "details": details,
        }
    )


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cifar_archive_path(data_cfg: Dict[str, Any], dataset: str) -> Path:
    if data_cfg.get("archive_path"):
        return Path(data_cfg["archive_path"])
    filename = "cifar-10-python.tar.gz" if dataset == "cifar10" else "cifar-100-python.tar.gz"
    return Path(data_cfg.get("raw_data_dir", ".data/cifar_raw")) / filename


def _expected_cifar_md5(data_cfg: Dict[str, Any], dataset: str) -> str:
    if data_cfg.get("archive_md5"):
        return str(data_cfg["archive_md5"])
    return _CIFAR10_MD5 if dataset == "cifar10" else _CIFAR100_MD5


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


def _check_cifar_direct(
    checks: List[Dict[str, Any]],
    config: Dict[str, Any],
    config_path: str,
    download_cifar: bool,
) -> None:
    data_cfg = config["data"]
    dataset = data_cfg["name"]
    archive = _cifar_archive_path(data_cfg, dataset)
    expected_md5 = _expected_cifar_md5(data_cfg, dataset)

    if not archive.exists() and download_cifar:
        loader = load_cifar10_direct if dataset == "cifar10" else load_cifar100_direct
        default_urls = _CIFAR10_URLS if dataset == "cifar10" else _CIFAR100_URLS
        loader(
            data_dir=data_cfg.get("raw_data_dir", ".data/cifar_raw"),
            archive_path=data_cfg.get("archive_path"),
            download_urls=data_cfg.get("download_urls", default_urls),
            download_timeout_seconds=data_cfg.get("download_timeout_seconds", 60),
            archive_md5=expected_md5,
            hyperval_size=data_cfg["hyperval_size"],
            seed=config.get("seed", 0),
        )

    if not archive.exists():
        _check(
            checks,
            "cifar_archive",
            "error",
            "CIFAR archive is missing. Download it manually or rerun preflight with --download-cifar.",
            archive=str(archive),
            expected_md5=expected_md5,
            config=config_path,
        )
        return

    actual_md5 = file_md5(archive)
    _check(
        checks,
        "cifar_archive",
        "ok" if actual_md5.lower() == expected_md5.lower() else "error",
        "CIFAR archive checksum verified."
        if actual_md5.lower() == expected_md5.lower()
        else "CIFAR archive checksum mismatch.",
        archive=str(archive),
        expected_md5=expected_md5,
        actual_md5=actual_md5,
        config=config_path,
    )


def _check_mnist_direct(
    checks: List[Dict[str, Any]],
    config: Dict[str, Any],
    config_path: str,
    download_mnist: bool,
) -> None:
    data_cfg = config["data"]
    root = Path(data_cfg.get("raw_data_dir", ".data/mnist_raw"))
    if download_mnist:
        load_mnist_direct(
            data_dir=str(root),
            download_mirrors=data_cfg.get("download_mirrors", _MNIST_MIRRORS),
            download_timeout_seconds=data_cfg.get("download_timeout_seconds", 60),
            resource_md5s=data_cfg.get("resource_md5s"),
            hyperval_size=data_cfg["hyperval_size"],
            seed=config.get("seed", 0),
        )

    resources = []
    missing = []
    mismatched = []
    resource_md5s = data_cfg.get("resource_md5s", _MNIST_RESOURCES)
    for filename, expected_md5 in resource_md5s.items():
        path = root / filename
        detail = {
            "filename": filename,
            "path": str(path),
            "expected_md5": expected_md5,
            "actual_md5": None,
            "exists": path.exists(),
        }
        if not path.exists():
            missing.append(filename)
        else:
            actual_md5 = file_md5(path)
            detail["actual_md5"] = actual_md5
            if actual_md5.lower() != expected_md5.lower():
                mismatched.append(filename)
        resources.append(detail)

    status = "ok" if not missing and not mismatched else "error"
    _check(
        checks,
        "mnist_resources",
        status,
        "MNIST IDX gzip resources verified."
        if status == "ok"
        else "MNIST IDX gzip resources are missing or have checksum mismatches.",
        config=config_path,
        raw_data_dir=str(root),
        missing=missing,
        mismatched=mismatched,
        resources=resources,
    )


def _check_tfds(checks: List[Dict[str, Any]], data_cfg: Dict[str, Any]) -> None:
    tensorflow_ok = _module_available("tensorflow")
    tfds_ok = _module_available("tensorflow_datasets")
    status = "ok" if tensorflow_ok and tfds_ok else "error"
    _check(
        checks,
        "tfds_dependencies",
        status,
        "TensorFlow and TensorFlow Datasets are importable."
        if status == "ok"
        else "TFDS-backed datasets need TensorFlow and tensorflow-datasets installed.",
        tensorflow=tensorflow_ok,
        tensorflow_datasets=tfds_ok,
        data_dir=data_cfg.get("data_dir"),
    )


def preflight_config(
    config_path: str,
    download_cifar: bool = False,
    download_mnist: bool = False,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    config = load_config(config_path)
    try:
        validate_config(config)
        _check(checks, "config_schema", "ok", "Training config schema is valid.")
    except Exception as exc:  # noqa: BLE001
        _check(checks, "config_schema", "error", "Training config schema is invalid.", error=str(exc))
        return {
            "config": config_path,
            "status": _overall_status(checks),
            "checks": checks,
        }

    data_cfg = config["data"]
    dataset = data_cfg["name"]
    train_examples = _estimate_train_examples(config)
    test_examples = _estimate_test_examples(config)
    classifier_steps = _steps_per_epoch(config, "classifier", train_examples)
    retrain_steps = _steps_per_epoch(config, "retrain", train_examples)
    _check(
        checks,
        "training_scale",
        "ok",
        "Training scale estimated from config.",
        dataset=dataset,
        train_examples=train_examples,
        classifier_steps_per_epoch=classifier_steps,
        classifier_total_steps=(
            classifier_steps * int(config["classifier"]["epochs"]) if classifier_steps is not None else None
        ),
        classifier_drop_last=_drop_last_for_section(config, "classifier"),
        augnet_steps=int(config["augnet"]["steps"]),
        retrain_steps_per_epoch=retrain_steps,
        retrain_total_steps=(
            retrain_steps * int(config["retrain"]["epochs"]) if retrain_steps is not None else None
        ),
        retrain_drop_last=_drop_last_for_section(config, "retrain"),
    )
    eval_batches = int(config.get("eval_batches", 0))
    eval_batch_size = int(config["classifier"]["batch_size"])
    expected_eval_batches = (
        (test_examples + eval_batch_size - 1) // eval_batch_size if test_examples is not None else None
    )
    if expected_eval_batches is not None:
        exact_full_eval = eval_batches == expected_eval_batches
        subset_debug_eval = data_cfg.get("max_test_size") is not None and eval_batches <= expected_eval_batches
        if exact_full_eval:
            eval_message = "Evaluation batches cover the test/validation split exactly once."
        elif subset_debug_eval:
            eval_message = "Subset/debug config evaluates the configured number of test batches."
        else:
            eval_message = "Evaluation batches do not cover the full test/validation split exactly once."
        _check(
            checks,
            "evaluation_coverage",
            "ok" if exact_full_eval or subset_debug_eval else "error",
            eval_message,
            dataset=dataset,
            configured_eval_batches=eval_batches,
            expected_eval_batches=expected_eval_batches,
            eval_batch_size=eval_batch_size,
            test_examples=test_examples,
            exact_full_eval=exact_full_eval,
            subset_debug_eval=subset_debug_eval,
        )

    source = data_cfg.get("source", "tfds" if dataset == "imagenet" else "direct")
    if dataset in ("cifar10", "cifar100") and source == "direct":
        _check_cifar_direct(checks, config, config_path, download_cifar=download_cifar)
    elif dataset == "mnist" and source == "direct":
        _check_mnist_direct(checks, config, config_path, download_mnist=download_mnist)
    elif source == "tfds":
        _check_tfds(checks, data_cfg)

    if dataset == "imagenet":
        info = imagenet_stream_info(data_cfg)
        _check(
            checks,
            "imagenet_stream",
            "ok",
            "ImageNet streaming splits are resolved.",
            input_shape=list(info.input_shape),
            train_examples=info.train_examples,
            hyperval_examples=info.hyperval_examples,
            validation_examples=info.validation_examples,
            train_split=info.train_split,
            hyperval_split=info.hyperval_split,
            validation_split=info.validation_split,
        )
    if dataset == "imagenet" and int(data_cfg.get("hyperval_size", 0)) != 50_000:
        _check(
            checks,
            "imagenet_hyperval_size",
            "warning",
            "Paper ImageNet experiments use a 50,000-sample hyper-validation split.",
            configured=data_cfg.get("hyperval_size"),
            expected=50_000,
        )
    if dataset == "imagenet" and int(data_cfg.get("validation_examples", IMAGENET_VALIDATION_EXAMPLES)) != 50_000:
        _check(
            checks,
            "imagenet_validation_size",
            "warning",
            "Paper ImageNet validation uses 50,000 validation examples.",
            configured=data_cfg.get("validation_examples", IMAGENET_VALIDATION_EXAMPLES),
            expected=50_000,
        )

    return {
        "config": config_path,
        "status": _overall_status(checks),
        "checks": checks,
    }


def config_paths_from_targets(
    targets_path: str,
    target_ids: Iterable[str] | None = None,
    tables: Iterable[Any] | None = None,
    datasets: Iterable[str] | None = None,
) -> List[str]:
    targets = filter_paper_targets(
        load_paper_targets(targets_path),
        target_ids=target_ids,
        tables=tables,
        datasets=datasets,
    )
    return unique_target_configs(targets)


def config_paths_from_suite(suite_dir: str | Path) -> List[str]:
    suite = Path(suite_dir)
    plan_path = suite / "suite_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        return [item["config"] for item in plan]
    return [str(path) for path in sorted((suite / "configs").glob("*.yaml"))]


def preflight_configs(
    config_paths: Iterable[str],
    download_cifar: bool = False,
    download_mnist: bool = False,
) -> List[Dict[str, Any]]:
    return [
        preflight_config(
            path,
            download_cifar=download_cifar,
            download_mnist=download_mnist,
        )
        for path in config_paths
    ]


def write_outputs(report: Dict[str, Any], output: str | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--target-id", nargs="*", default=None)
    parser.add_argument("--table", nargs="*", default=None, help="Filter paper targets by table number, e.g. 1 2.")
    parser.add_argument("--dataset", nargs="*", default=None, help="Filter paper targets by dataset name.")
    parser.add_argument("--suite-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--download-cifar", action="store_true")
    parser.add_argument("--download-mnist", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.configs:
        config_paths = args.configs
    elif args.suite_dir:
        config_paths = config_paths_from_suite(args.suite_dir)
    else:
        config_paths = config_paths_from_targets(
            args.targets,
            target_ids=args.target_id,
            tables=args.table,
            datasets=args.dataset,
        )

    rows = preflight_configs(
        config_paths,
        download_cifar=args.download_cifar,
        download_mnist=args.download_mnist,
    )
    summary: Dict[str, int] = {}
    for row in rows:
        summary[row["status"]] = summary.get(row["status"], 0) + 1

    report = {
        "summary": summary,
        "configs": rows,
    }
    write_outputs(report, args.output)
    print(json.dumps(summary, sort_keys=True))
    if args.output:
        print(f"wrote {args.output}")

    if args.strict and any(row["status"] != "ok" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
