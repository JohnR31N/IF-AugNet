from pathlib import Path
import argparse
import csv
import json
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import flatten_metric_record, latest_by_stage, load_jsonl_metrics


def _metric_series(records: List[Dict], metric_name: str) -> List[Tuple[float, float]]:
    points = []
    for index, record in enumerate(records):
        metrics = record.get("metrics", {})
        if metric_name in metrics:
            step = record.get("step", index)
            points.append((float(step), float(metrics[metric_name])))
    return points


def _polyline(points: List[Tuple[float, float]], x0: int, y0: int, width: int, height: int) -> str:
    if not points:
        return ""

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = xmax - xmin if xmax != xmin else 1.0
    yspan = ymax - ymin if ymax != ymin else 1.0

    coords = []
    for x, y in points:
        px = x0 + ((x - xmin) / xspan) * width
        py = y0 + height - ((y - ymin) / yspan) * height
        coords.append(f"{px:.1f},{py:.1f}")
    return " ".join(coords)


def write_curves_svg(records: List[Dict], path: str, metrics: List[str]) -> None:
    panel_width = 420
    panel_height = 180
    margin = 48
    gap = 32
    width = panel_width + 2 * margin
    height = len(metrics) * (panel_height + gap) + margin

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#111827}.axis{stroke:#9ca3af;stroke-width:1}.line{fill:none;stroke-width:2}</style>',
    ]

    for i, metric_name in enumerate(metrics):
        y = margin + i * (panel_height + gap)
        points = _metric_series(records, metric_name)
        parts.append(f'<text x="{margin}" y="{y - 14}">{metric_name}</text>')
        parts.append(
            f'<line class="axis" x1="{margin}" y1="{y + panel_height}" x2="{margin + panel_width}" y2="{y + panel_height}"/>'
        )
        parts.append(f'<line class="axis" x1="{margin}" y1="{y}" x2="{margin}" y2="{y + panel_height}"/>')
        polyline = _polyline(points, margin, y, panel_width, panel_height)
        if polyline:
            color = colors[i % len(colors)]
            parts.append(f'<polyline class="line" stroke="{color}" points="{polyline}"/>')

    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to metrics.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--plot-metrics",
        nargs="*",
        default=["loss", "accuracy", "estimated_val_loss_reduction"],
    )
    args = parser.parse_args()

    records = load_jsonl_metrics(args.metrics)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.metrics).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [flatten_metric_record(record) for record in records]
    fieldnames = sorted({field for row in rows for field in row})
    csv_path = output_dir / "metrics_flat.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        stage: flatten_metric_record(record)
        for stage, record in latest_by_stage(records).items()
    }
    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    svg_path = output_dir / "metrics_curves.svg"
    write_curves_svg(records, str(svg_path), args.plot_metrics)

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
