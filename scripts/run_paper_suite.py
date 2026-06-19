from pathlib import Path
import argparse
import json
import subprocess
import sys
from typing import Any, Dict, Iterable, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_to_paper import (
    DEFAULT_SEEDS,
    DEFAULT_TARGETS,
    filter_paper_targets,
    load_paper_targets,
    suite_config_path,
)
from utils import load_config, output_dir_from_arg, validate_config


def targets_by_config(targets: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    seen = set()
    for target in targets:
        config_path = target["config"]
        if config_path not in seen:
            seen.add(config_path)
            grouped[config_path] = []
        grouped[config_path].append(target)
    return grouped


def unique_target_configs(targets: Iterable[Dict[str, Any]]) -> List[str]:
    return list(targets_by_config(targets).keys())


def _target_summary(target: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": target["id"],
        "table": target.get("table"),
        "setting": target.get("setting", ""),
        "metric": target["metric"],
        "direction": target.get("direction", "max"),
        "target": target["target"],
        "units": target.get("units", "percent"),
    }


def _runner_for_config(config: Dict[str, Any]) -> str:
    return "scripts/train_imagenet_stream.py" if config["data"]["name"] == "imagenet" else "scripts/train.py"


def materialize_suite_configs(
    targets: Iterable[Dict[str, Any]],
    output_dir: str | Path,
    seeds: Iterable[int],
) -> List[Dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "configs").mkdir(parents=True, exist_ok=True)

    plan = []
    grouped_targets = targets_by_config(targets)
    for config_path, config_targets in grouped_targets.items():
        base_config = load_config(config_path)
        validate_config(base_config)
        stem = Path(config_path).stem
        target_summaries = [_target_summary(target) for target in config_targets]
        for seed in seeds:
            run_config = dict(base_config)
            run_config["seed"] = int(seed)
            run_config["checkpoint_dir"] = str(output / "runs" / stem / f"seed_{seed}")
            generated_config = suite_config_path(output, config_path, int(seed))
            generated_config.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
            plan.append(
                {
                    "base_config": config_path,
                    "config": str(generated_config),
                    "seed": int(seed),
                    "checkpoint_dir": run_config["checkpoint_dir"],
                    "dataset": run_config["data"]["name"],
                    "runner": _runner_for_config(run_config),
                    "target_ids": [target["id"] for target in config_targets],
                    "targets": target_summaries,
                }
            )
    return plan


def command_for_run(
    item: Dict[str, Any],
    stage: str,
    resume: bool,
    python_executable: str = sys.executable,
    stop_after_steps: int | None = None,
) -> List[str]:
    command = [
        python_executable,
        item["runner"],
        "--config",
        item["config"],
        "--stage",
        stage,
    ]
    if resume:
        command.append("--resume")
    if stop_after_steps is not None:
        command.extend(["--stop-after-steps", str(stop_after_steps)])
    return command


def write_plan(plan: List[Dict[str, Any]], output_dir: str | Path) -> None:
    path = Path(output_dir) / "suite_plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", default="runs/paper_suite")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--stage",
        choices=("classifier", "pretrain_augnet", "augnet", "retrain", "all"),
        default="all",
    )
    parser.add_argument("--target-id", nargs="*", default=None)
    parser.add_argument("--table", nargs="*", default=None, help="Filter paper targets by table number, e.g. 1 2.")
    parser.add_argument("--dataset", nargs="*", default=None, help="Filter paper targets by dataset name.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        default=None,
        help="Pass through to each runner to execute only a chunk of one selected stage.",
    )
    args = parser.parse_args()
    if args.stop_after_steps is not None and args.stage == "all":
        parser.error("--stop-after-steps requires selecting one stage, not --stage all.")

    targets = filter_paper_targets(
        load_paper_targets(args.targets),
        target_ids=args.target_id,
        tables=args.table,
        datasets=args.dataset,
    )

    try:
        output_dir = output_dir_from_arg(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))

    plan = materialize_suite_configs(targets, output_dir, args.seeds)
    for item in plan:
        item["command"] = command_for_run(
            item,
            args.stage,
            args.resume,
            stop_after_steps=args.stop_after_steps,
        )
    write_plan(plan, output_dir)

    print(f"wrote {output_dir / 'suite_plan.json'}")
    print(json.dumps({"runs": len(plan), "seeds": args.seeds, "dry_run": args.dry_run}, sort_keys=True))
    if args.dry_run:
        return

    for item in plan:
        subprocess.run(item["command"], check=True)


if __name__ == "__main__":
    main()
