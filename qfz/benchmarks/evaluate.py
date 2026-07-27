"""Summarize benchmark result JSONs into a comparison table.

Usage::

    python -m qfz.benchmarks.evaluate --results results

Reads every ``*.json`` record written by ``qfz.benchmarks.train`` /
``run_all`` and prints a markdown table comparing models per dataset.
Runs sharing a (dataset, variant-or-model) key — e.g. the same model
trained with several seeds — are aggregated into mean +/- std rows.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_records(results_dir: str) -> list:
    """Load all benchmark records from a directory (excluding summaries)."""
    records = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name.startswith("summary"):
            continue
        records.append(json.loads(path.read_text()))
    return records


def format_table(records: list) -> str:
    """Render per-run benchmark records as a markdown table."""
    header = ("| dataset | model | test acc | trainable params | "
              "train time (s) | inference (ms/img) |")
    rule = "|---|---|---|---|---|---|"
    lines = [header, rule]
    for r in sorted(records, key=lambda r: (r["config"]["dataset"], r["config"]["model"])):
        c, m = r["config"], r["metrics"]
        lines.append(
            f"| {c['dataset']} | {c.get('variant') or c['model']} | {m['test_accuracy']:.4f} "
            f"| {m['parameters']['trainable']:,} | {m['train_time_s']:.1f} "
            f"| {m['inference']['ms_per_image']:.2f} |")
    return "\n".join(lines)


def aggregate_records(records: list) -> list:
    """Group records by (dataset, variant-or-model) and aggregate over seeds.

    Returns:
        One dict per group with mean/std test accuracy, mean timings, the
        trainable parameter count, and the number of runs (``n_seeds``).
    """
    groups = defaultdict(list)
    for r in records:
        c = r["config"]
        groups[(c["dataset"], c.get("variant") or c["model"])].append(r)

    rows = []
    for (dataset, model), rs in sorted(groups.items()):
        accs = [r["metrics"]["test_accuracy"] for r in rs]
        rows.append({
            "dataset": dataset,
            "model": model,
            "n_seeds": len(rs),
            "seeds": sorted(r["config"]["seed"] for r in rs),
            "test_accuracy_mean": statistics.mean(accs),
            "test_accuracy_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "trainable_params": rs[0]["metrics"]["parameters"]["trainable"],
            "train_time_s_mean": statistics.mean(
                r["metrics"]["train_time_s"] for r in rs),
            "inference_ms_per_image_mean": statistics.mean(
                r["metrics"]["inference"]["ms_per_image"] for r in rs),
        })
    return rows


def format_aggregate_table(rows: list) -> str:
    """Render aggregated rows (from :func:`aggregate_records`) as markdown."""
    header = ("| dataset | model | test acc (mean ± std) | n | "
              "trainable params | train time (s) | inference (ms/img) |")
    rule = "|---|---|---|---|---|---|---|"
    lines = [header, rule]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} "
            f"| {row['test_accuracy_mean']:.4f} ± {row['test_accuracy_std']:.4f} "
            f"| {row['n_seeds']} | {row['trainable_params']:,} "
            f"| {row['train_time_s_mean']:.1f} "
            f"| {row['inference_ms_per_image_mean']:.2f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--per-run", action="store_true",
                        help="print one row per run instead of aggregating over seeds")
    args = parser.parse_args()

    records = load_records(args.results)
    if not records:
        print(f"No benchmark records found in '{args.results}'.")
        return
    if args.per_run:
        print(format_table(records))
    else:
        print(format_aggregate_table(aggregate_records(records)))


if __name__ == "__main__":
    main()
