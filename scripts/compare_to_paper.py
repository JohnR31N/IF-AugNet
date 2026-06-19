from pathlib import Path
import argparse
import csv
import json
import statistics
import sys
from typing import Any, Dict, Iterable, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_results import collect_config_result
from utils import load_config, output_dir_from_arg


DEFAULT_TARGETS = "configs/paper_targets.yaml"
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def _percent(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


def metric_value(row: Dict[str, Any], metric: str) -> float | None:
    mapping = {
        "baseline_error_percent": lambda: row.get("baseline_error"),
        "augnet_error_percent": lambda: row.get("augnet_error"),
        "baseline_top1_percent": lambda: _percent(row.get("baseline_accuracy")),
        "augnet_top1_percent": lambda: _percent(row.get("augnet_accuracy")),
        "baseline_top5_percent": lambda: _percent(row.get("baseline_top5_accuracy")),
        "augnet_top5_percent": lambda: _percent(row.get("augnet_top5_accuracy")),
    }
    if metric not in mapping:
        raise ValueError(f"Unsupported target metric: {metric}")
    value = mapping[metric]()
    return None if value is None else float(value)


def load_paper_targets(path: str = DEFAULT_TARGETS) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    targets = payload.get("targets", []) if isinstance(payload, dict) else []
    if not targets:
        raise ValueError(f"No targets found in {path}")
    return list(targets)


def _normalize_table(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("table"):
        text = text[len("table") :].strip()
    return text


def _target_dataset(target: Dict[str, Any]) -> str:
    return str(load_config(target["config"])["data"]["name"]).lower()


def filter_paper_targets(
    targets: Iterable[Dict[str, Any]],
    target_ids: Iterable[str] | None = None,
    tables: Iterable[Any] | None = None,
    datasets: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    selected = list(targets)
    if target_ids:
        requested = set(target_ids)
        selected = [target for target in selected if target["id"] in requested]
        missing = sorted(requested.difference(target["id"] for target in selected))
        if missing:
            raise ValueError(f"Unknown target id(s): {', '.join(missing)}")
    if tables:
        requested_tables = {_normalize_table(table) for table in tables}
        selected = [target for target in selected if _normalize_table(target.get("table")) in requested_tables]
    if datasets:
        requested_datasets = {str(dataset).lower() for dataset in datasets}
        selected = [target for target in selected if _target_dataset(target) in requested_datasets]
    if not selected:
        raise ValueError("Target filters matched no paper targets.")
    return selected


def evaluate_target(
    target: Dict[str, Any],
    tolerance: float = 0.0,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    config_path = target["config"]
    result_row = row if row is not None else collect_config_result(config_path)
    actual = metric_value(result_row, target["metric"])
    expected = float(target["target"])
    direction = target.get("direction", "max")

    if actual is None:
        status = "missing_metric"
        passed = False
        delta = None
    elif direction == "max":
        passed = actual <= expected + tolerance
        delta = actual - expected
        status = "pass" if passed else "fail"
    elif direction == "min":
        passed = actual >= expected - tolerance
        delta = actual - expected
        status = "pass" if passed else "fail"
    else:
        raise ValueError(f"Unsupported direction for {target['id']}: {direction}")

    return {
        "id": target["id"],
        "table": target.get("table"),
        "setting": target.get("setting", ""),
        "config": config_path,
        "metric": target["metric"],
        "direction": direction,
        "target": expected,
        "actual": actual,
        "actual_std": None,
        "n": 1 if actual is not None else 0,
        "expected_runs": 1,
        "missing_runs": 0 if actual is not None else 1,
        "delta": delta,
        "tolerance": tolerance,
        "units": target.get("units", "percent"),
        "status": status,
        "passed": passed,
        "run_status": result_row.get("status"),
        "run_dir": result_row.get("run_dir"),
        "per_seed": [],
    }


def suite_config_path(suite_dir: str | Path, config_path: str, seed: int) -> Path:
    return Path(suite_dir) / "configs" / f"{Path(config_path).stem}_seed{seed}.yaml"


def evaluate_suite_target(
    target: Dict[str, Any],
    suite_dir: str | Path,
    seeds: Iterable[int],
    tolerance: float = 0.0,
    require_all_seeds: bool = True,
) -> Dict[str, Any]:
    per_seed = []
    values = []
    for seed in seeds:
        generated_config = suite_config_path(suite_dir, target["config"], seed)
        if not generated_config.exists():
            per_seed.append(
                {
                    "seed": seed,
                    "config": str(generated_config),
                    "actual": None,
                    "status": "missing_config",
                    "run_status": "missing_config",
                    "run_dir": "",
                }
            )
            continue
        row = collect_config_result(str(generated_config))
        actual = metric_value(row, target["metric"])
        if actual is not None:
            values.append(actual)
        per_seed.append(
            {
                "seed": seed,
                "config": str(generated_config),
                "actual": actual,
                "status": "ok" if actual is not None else "missing_metric",
                "run_status": row.get("status"),
                "run_dir": row.get("run_dir"),
            }
        )

    expected = float(target["target"])
    direction = target.get("direction", "max")
    missing_runs = len(per_seed) - len(values)
    actual = statistics.mean(values) if values else None
    actual_std = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)

    if actual is None:
        status = "missing_metric"
        passed = False
        delta = None
    elif require_all_seeds and missing_runs:
        status = "missing_seed_metrics"
        passed = False
        delta = actual - expected
    elif direction == "max":
        passed = actual <= expected + tolerance
        delta = actual - expected
        status = "pass" if passed else "fail"
    elif direction == "min":
        passed = actual >= expected - tolerance
        delta = actual - expected
        status = "pass" if passed else "fail"
    else:
        raise ValueError(f"Unsupported direction for {target['id']}: {direction}")

    return {
        "id": target["id"],
        "table": target.get("table"),
        "setting": target.get("setting", ""),
        "config": target["config"],
        "metric": target["metric"],
        "direction": direction,
        "target": expected,
        "actual": actual,
        "actual_std": actual_std,
        "n": len(values),
        "expected_runs": len(per_seed),
        "missing_runs": missing_runs,
        "delta": delta,
        "tolerance": tolerance,
        "units": target.get("units", "percent"),
        "status": status,
        "passed": passed,
        "run_status": f"{len(values)}/{len(per_seed)} seed metrics",
        "run_dir": str(Path(suite_dir)),
        "per_seed": per_seed,
    }


def compare_targets(
    targets: Iterable[Dict[str, Any]],
    tolerance: float = 0.0,
) -> List[Dict[str, Any]]:
    rows_by_config: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    for target in targets:
        config_path = target["config"]
        if config_path not in rows_by_config:
            rows_by_config[config_path] = collect_config_result(config_path)
        results.append(evaluate_target(target, tolerance=tolerance, row=rows_by_config[config_path]))
    return results


def compare_suite_targets(
    targets: Iterable[Dict[str, Any]],
    suite_dir: str | Path,
    seeds: Iterable[int],
    tolerance: float = 0.0,
    require_all_seeds: bool = True,
) -> List[Dict[str, Any]]:
    return [
        evaluate_suite_target(
            target,
            suite_dir=suite_dir,
            seeds=seeds,
            tolerance=tolerance,
            require_all_seeds=require_all_seeds,
        )
        for target in targets
    ]


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "table",
        "setting",
        "config",
        "metric",
        "direction",
        "target",
        "actual",
        "actual_std",
        "n",
        "expected_runs",
        "missing_runs",
        "delta",
        "tolerance",
        "units",
        "status",
        "passed",
        "run_status",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("table", "Table"),
        ("setting", "Setting"),
        ("metric", "Metric"),
        ("target", "Target"),
        ("actual", "Actual"),
        ("actual_std", "Std"),
        ("n", "N"),
        ("status", "Status"),
        ("run_status", "Run status"),
    ]
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[key]) for key, _ in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "paper_comparison.csv")
    write_markdown(rows, output_dir / "paper_comparison.md")
    (output_dir / "paper_comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", default="runs/paper_compare")
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument(
        "--suite-dir",
        default=None,
        help="Optional run_paper_suite.py output directory containing generated per-seed configs.",
    )
    parser.add_argument("--target-id", nargs="*", default=None)
    parser.add_argument("--table", nargs="*", default=None, help="Filter paper targets by table number, e.g. 1 2.")
    parser.add_argument("--dataset", nargs="*", default=None, help="Filter paper targets by dataset name.")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--allow-partial-seeds",
        action="store_true",
        help="When --suite-dir is used, compare the mean of available seeds instead of requiring every requested seed.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    targets = filter_paper_targets(
        load_paper_targets(args.targets),
        target_ids=args.target_id,
        tables=args.table,
        datasets=args.dataset,
    )
    if args.suite_dir:
        rows = compare_suite_targets(
            targets,
            suite_dir=args.suite_dir,
            seeds=args.seeds,
            tolerance=args.tolerance,
            require_all_seeds=not args.allow_partial_seeds,
        )
    else:
        rows = compare_targets(targets, tolerance=args.tolerance)
    try:
        output_dir = output_dir_from_arg(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    write_outputs(rows, output_dir)

    summary: Dict[str, int] = {}
    for row in rows:
        summary[row["status"]] = summary.get(row["status"], 0) + 1

    print(f"wrote {output_dir / 'paper_comparison.csv'}")
    print(f"wrote {output_dir / 'paper_comparison.md'}")
    print(f"wrote {output_dir / 'paper_comparison.json'}")
    print(json.dumps(summary, sort_keys=True))

    if args.strict and any(not row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
