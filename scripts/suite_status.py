from pathlib import Path
import argparse
import csv
import json
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_results import collect_config_result
from utils import load_pickle, output_dir_from_arg


STAGES = ("classifier", "pretrain_augnet", "augnet", "retrain")


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _stage_progress(run_dir: Path, stage: str) -> Dict[str, Any]:
    progress_path = run_dir / f"{stage}_progress.pkl"
    if not progress_path.exists():
        return {
            f"{stage}_completed": False,
            f"{stage}_next_step": None,
            f"{stage}_total_steps": None,
            f"{stage}_progress_status": "missing",
        }
    try:
        progress = load_pickle(str(progress_path))
    except Exception as exc:  # noqa: BLE001
        return {
            f"{stage}_completed": False,
            f"{stage}_next_step": None,
            f"{stage}_total_steps": None,
            f"{stage}_progress_status": f"error: {exc}",
        }
    return {
        f"{stage}_completed": bool(progress.get("completed", False)),
        f"{stage}_next_step": progress.get("next_step"),
        f"{stage}_total_steps": progress.get("total_steps"),
        f"{stage}_progress_status": "ok",
    }


def _run_state(row: Dict[str, Any], stage_details: Dict[str, Any], config_exists: bool) -> str:
    if not config_exists:
        return "missing_config"
    if row["status"] == "ok" and all(bool(row.get(f"{stage}_completed")) for stage in STAGES):
        return "complete"
    if any(stage_details.get(f"{stage}_progress_status") == "ok" for stage in STAGES):
        return "in_progress"
    if row["status"] == "missing_metrics":
        return "pending"
    return str(row["status"])


def load_suite_plan(suite_dir: str | Path) -> List[Dict[str, Any]]:
    path = Path(suite_dir) / "suite_plan.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing suite plan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def collect_suite_status(suite_dir: str | Path) -> List[Dict[str, Any]]:
    rows = []
    for index, item in enumerate(load_suite_plan(suite_dir)):
        config_path = Path(item["config"])
        config_exists = config_path.exists()
        row = collect_config_result(str(config_path)) if config_exists else {
            "status": "missing_config",
            "run_dir": item.get("checkpoint_dir", ""),
        }
        run_dir = Path(row.get("run_dir") or item.get("checkpoint_dir", ""))
        stage_details: Dict[str, Any] = {}
        for stage in STAGES:
            stage_details.update(_stage_progress(run_dir, stage))

        targets = item.get("targets", [])
        status = {
            "index": index,
            "dataset": item.get("dataset"),
            "seed": item.get("seed"),
            "base_config": item.get("base_config"),
            "config": item.get("config"),
            "run_dir": str(run_dir),
            "target_ids": ",".join(item.get("target_ids", [])),
            "target_metrics": ",".join(target.get("metric", "") for target in targets),
            "run_state": _run_state(row, stage_details, config_exists),
            "metric_status": row.get("status"),
            "baseline_error": row.get("baseline_error"),
            "augnet_error": row.get("augnet_error"),
            "baseline_top1": (
                row.get("baseline_accuracy") * 100.0
                if row.get("baseline_accuracy") is not None
                else None
            ),
            "augnet_top1": (
                row.get("augnet_accuracy") * 100.0
                if row.get("augnet_accuracy") is not None
                else None
            ),
            "baseline_top5": (
                row.get("baseline_top5_accuracy") * 100.0
                if row.get("baseline_top5_accuracy") is not None
                else None
            ),
            "augnet_top5": (
                row.get("augnet_top5_accuracy") * 100.0
                if row.get("augnet_top5_accuracy") is not None
                else None
            ),
            "s_test_residual_mean": row.get("s_test_residual_mean"),
        }
        status.update(stage_details)
        rows.append(status)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for row in rows:
        state = row["run_state"]
        summary[state] = summary.get(state, 0) + 1
    return summary


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "dataset",
        "seed",
        "base_config",
        "config",
        "run_dir",
        "target_ids",
        "target_metrics",
        "run_state",
        "metric_status",
        "baseline_error",
        "augnet_error",
        "baseline_top1",
        "augnet_top1",
        "baseline_top5",
        "augnet_top5",
        "s_test_residual_mean",
    ]
    for stage in STAGES:
        fieldnames.extend(
            [
                f"{stage}_completed",
                f"{stage}_next_step",
                f"{stage}_total_steps",
                f"{stage}_progress_status",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("index", "#"),
        ("dataset", "Dataset"),
        ("seed", "Seed"),
        ("target_ids", "Targets"),
        ("run_state", "State"),
        ("metric_status", "Metrics"),
        ("classifier_completed", "Classifier"),
        ("pretrain_augnet_completed", "Pretrain"),
        ("augnet_completed", "AugNet"),
        ("retrain_completed", "Retrain"),
        ("augnet_error", "AugNet error"),
        ("augnet_top1", "AugNet Top-1"),
        ("augnet_top5", "AugNet Top-5"),
    ]
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(key)) for key, _ in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "suite_status.csv")
    write_markdown(rows, output_dir / "suite_status.md")
    (output_dir / "suite_status.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "suite_status_summary.json").write_text(
        json.dumps(summarize(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--suite-dir", default="runs/paper_suite")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    try:
        rows = collect_suite_status(args.suite_dir)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        parser.error(str(exc))
    output_dir = None
    if args.output_dir:
        try:
            output_dir = output_dir_from_arg(args.output_dir)
        except ValueError as exc:
            parser.error(str(exc))
        write_outputs(rows, output_dir)

    summary = summarize(rows)
    print(json.dumps(summary, sort_keys=True))
    if output_dir:
        print(f"wrote {output_dir / 'suite_status.csv'}")
        print(f"wrote {output_dir / 'suite_status.md'}")
        print(f"wrote {output_dir / 'suite_status.json'}")

    if args.strict and any(row["run_state"] != "complete" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
