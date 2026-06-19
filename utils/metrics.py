import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class JsonlMetricLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        stage: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        record = {
            "stage": stage,
            "metrics": metrics,
        }
        if step is not None:
            record["step"] = step
        if epoch is not None:
            record["epoch"] = epoch

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def load_jsonl_metrics(path: str) -> List[Dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def flatten_metric_record(record: Dict) -> Dict[str, object]:
    flat = {
        "stage": record["stage"],
        "step": record.get("step", ""),
        "epoch": record.get("epoch", ""),
    }
    for key, value in record.get("metrics", {}).items():
        flat[f"metric.{key}"] = value
    return flat


def latest_by_stage(records: Iterable[Dict]) -> Dict[str, Dict]:
    latest = {}
    for record in records:
        latest[record["stage"]] = record
    return latest
