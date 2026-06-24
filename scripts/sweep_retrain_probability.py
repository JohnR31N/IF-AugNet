from pathlib import Path
import argparse
import csv
import json
import shutil
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from utils import load_config, validate_config


def _probability_label(value: float) -> str:
    """Return a filesystem-friendly label such as p003 for 0.03."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "p" + text.replace(".", "")


def _write_config(base_config_path: Path, output_config_path: Path, output_run_dir: Path, probability: float, learned_aug_input: str) -> None:
    """Write a generated retrain-only config with a new checkpoint directory."""
    cfg: dict[str, Any] = load_config(str(base_config_path))
    cfg["checkpoint_dir"] = str(output_run_dir)
    cfg["retrain"]["learned_aug_probability"] = float(probability)
    cfg["retrain"]["learned_aug_input"] = learned_aug_input
    validate_config(cfg)
    cfg["_sweep"] = {
        "base_config": str(base_config_path),
        "learned_aug_probability": float(probability),
        "learned_aug_input": learned_aug_input,
    }
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _copy_run_dir(source_run_dir: Path, output_run_dir: Path, overwrite: bool) -> None:
    """Copy existing classifier/AugNet checkpoints into a sweep run directory."""
    if not source_run_dir.exists():
        raise FileNotFoundError(f"Source run directory does not exist: {source_run_dir}")
    if output_run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output run directory exists; pass --overwrite: {output_run_dir}")
        shutil.rmtree(output_run_dir)
    shutil.copytree(source_run_dir, output_run_dir)


def _run_command(command: list[str], dry_run: bool) -> None:
    """Print and optionally execute one subprocess command."""
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean for non-empty numeric lists."""
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    """Return population standard deviation for compact sweep summaries."""
    if not values:
        return None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _format_value(value: Any) -> str:
    """Format summary cells for Markdown output."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_apply_best_script(rows: list[dict[str, Any]], best_row: dict[str, Any] | None, output_dir: Path) -> Path | None:
    """Write a helper that applies a positive best sweep value to base configs."""
    if best_row is None or not best_row.get("beats_baseline"):
        return None

    best_probability = float(best_row["probability"])
    base_configs: list[str] = []
    learned_aug_inputs: set[str] = set()
    for row in rows:
        cfg = load_config(row["config"])
        probability = float(cfg["retrain"].get("learned_aug_probability", 1.0))
        if probability != best_probability:
            continue
        sweep_meta = cfg.get("_sweep", {})
        base_config = sweep_meta.get("base_config")
        if base_config and base_config not in base_configs:
            base_configs.append(base_config)
        learned_aug_input = cfg["retrain"].get("learned_aug_input")
        if learned_aug_input:
            learned_aug_inputs.add(str(learned_aug_input))

    if not base_configs:
        return None

    learned_aug_input = sorted(learned_aug_inputs)[0] if len(learned_aug_inputs) == 1 else None
    script_path = output_dir / "apply_best_probability.py"
    script = f"""from pathlib import Path
import yaml

CONFIG_PATHS = {json.dumps(base_configs, indent=2)}
PROBABILITY = {best_probability!r}
LEARNED_AUG_INPUT = {learned_aug_input!r}

for config_path in CONFIG_PATHS:
    path = Path(config_path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg.setdefault("retrain", {{}})["learned_aug_probability"] = PROBABILITY
    if LEARNED_AUG_INPUT is not None:
        cfg["retrain"]["learned_aug_input"] = LEARNED_AUG_INPUT
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"updated {{path}}")
"""
    script_path.write_text(script, encoding="utf-8")
    return script_path


def _summarize_results(results_json: Path, output_dir: Path) -> None:
    """Aggregate collected rows by learned augmentation probability."""
    if not results_json.exists():
        raise FileNotFoundError(f"Cannot summarize missing results file: {results_json}")
    rows = json.loads(results_json.read_text(encoding="utf-8"))
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        cfg = load_config(row["config"])
        probability = float(cfg["retrain"].get("learned_aug_probability", 1.0))
        grouped.setdefault(probability, []).append(row)

    summary_rows = []
    for probability, probability_rows in sorted(grouped.items()):
        ok_rows = [row for row in probability_rows if row.get("status") == "ok"]
        baseline_errors = [
            float(row["baseline_error"])
            for row in ok_rows
            if row.get("baseline_error") is not None
        ]
        augnet_errors = [
            float(row["augnet_error"])
            for row in ok_rows
            if row.get("augnet_error") is not None
        ]
        reductions = [
            float(row["error_reduction"])
            for row in ok_rows
            if row.get("error_reduction") is not None
        ]
        summary_rows.append(
            {
                "probability": probability,
                "runs": len(probability_rows),
                "ok_runs": len(ok_rows),
                "baseline_error_mean": _mean(baseline_errors),
                "augnet_error_mean": _mean(augnet_errors),
                "error_reduction_mean": _mean(reductions),
                "error_reduction_std": _std(reductions),
                "beats_baseline": bool(reductions and _mean(reductions) > 0.0),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary_by_probability.csv"
    fieldnames = [
        "probability",
        "runs",
        "ok_runs",
        "baseline_error_mean",
        "augnet_error_mean",
        "error_reduction_mean",
        "error_reduction_std",
        "beats_baseline",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = output_dir / "summary_by_probability.json"
    json_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    comparable_rows = [
        row for row in summary_rows if row["error_reduction_mean"] is not None
    ]
    complete_rows = [
        row for row in comparable_rows if row["runs"] > 0 and row["ok_runs"] == row["runs"]
    ]
    best_candidates = complete_rows or comparable_rows
    best_row = (
        max(best_candidates, key=lambda row: row["error_reduction_mean"])
        if best_candidates
        else None
    )
    best_json_path = output_dir / "best_probability.json"
    best_json_path.write_text(
        json.dumps(best_row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    apply_script_path = _write_apply_best_script(rows, best_row, output_dir)

    markdown_path = output_dir / "summary_by_probability.md"
    columns = [
        ("probability", "p"),
        ("ok_runs", "OK runs"),
        ("baseline_error_mean", "Baseline error"),
        ("augnet_error_mean", "AugNet error"),
        ("error_reduction_mean", "Mean reduction"),
        ("error_reduction_std", "Reduction std"),
        ("beats_baseline", "Beats baseline"),
    ]
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join(_format_value(row[key]) for key, _ in columns) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best_markdown_path = output_dir / "best_probability.md"
    if best_row is None:
        best_markdown = "No completed probability runs were found.\n"
    else:
        best_markdown = "\n".join(
            [
                f"Best probability: `{_format_value(best_row['probability'])}`",
                f"Mean reduction: `{_format_value(best_row['error_reduction_mean'])}`",
                f"AugNet error: `{_format_value(best_row['augnet_error_mean'])}`",
                f"Baseline error: `{_format_value(best_row['baseline_error_mean'])}`",
                f"Beats baseline: `{best_row['beats_baseline']}`",
                f"Complete group: `{best_row['ok_runs'] == best_row['runs']}`",
                (
                    f"Apply command: `python {apply_script_path}`"
                    if apply_script_path is not None
                    else "Apply command: not generated unless the best complete group beats baseline and records base configs."
                ),
            ]
        ) + "\n"
    best_markdown_path.write_text(best_markdown, encoding="utf-8")

    print(f"Summary table: {markdown_path}", flush=True)
    print(f"Best probability: {best_markdown_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Base configs whose checkpoint_dir contains trained classifier and AugNet checkpoints.",
    )
    parser.add_argument(
        "--probabilities",
        nargs="+",
        type=float,
        help="Retrain learned_aug_probability values to sweep.",
    )
    parser.add_argument("--learned-aug-input", default="raw", choices=("raw", "baseline"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Reuse an existing results/results_table.json and regenerate probability summaries.",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.summarize_only:
        results_json = output_dir / "results" / "results_table.json"
        if args.dry_run:
            print(f"Would summarize {results_json} into {output_dir}", flush=True)
            return
        _summarize_results(output_dir / "results" / "results_table.json", output_dir)
        return
    if not args.configs:
        parser.error("--configs is required unless --summarize-only is used.")
    if not args.probabilities:
        parser.error("--probabilities is required unless --summarize-only is used.")

    generated_config_dir = output_dir / "generated_configs"
    generated_configs: list[Path] = []

    for probability in args.probabilities:
        probability_label = _probability_label(probability)
        for config_arg in args.configs:
            base_config_path = Path(config_arg)
            cfg = load_config(str(base_config_path))
            source_run_dir = Path(cfg["checkpoint_dir"])
            output_run_dir = output_dir / f"{source_run_dir.name}_{probability_label}_{args.learned_aug_input}"
            output_config_path = generated_config_dir / f"{base_config_path.stem}_{probability_label}_{args.learned_aug_input}.yaml"

            print(f"Preparing {output_config_path} -> {output_run_dir}", flush=True)
            if not args.dry_run:
                _copy_run_dir(source_run_dir, output_run_dir, overwrite=args.overwrite)
                _write_config(
                    base_config_path,
                    output_config_path,
                    output_run_dir,
                    probability,
                    args.learned_aug_input,
                )
            generated_configs.append(output_config_path)

            _run_command(
                [
                    args.python,
                    "scripts/train.py",
                    "--config",
                    str(output_config_path),
                    "--stage",
                    "retrain",
                ],
                dry_run=args.dry_run,
            )

    collect_output_dir = output_dir / "results"
    _run_command(
        [
            args.python,
            "scripts/collect_results.py",
            "--configs",
            *[str(path) for path in generated_configs],
            "--output-dir",
            str(collect_output_dir),
        ],
        dry_run=args.dry_run,
    )
    print(f"Results table: {collect_output_dir / 'results_table.md'}", flush=True)
    if not args.dry_run:
        _summarize_results(collect_output_dir / "results_table.json", output_dir)


if __name__ == "__main__":
    main()
