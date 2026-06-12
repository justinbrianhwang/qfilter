"""Summarize benchmark result JSONs into a comparison table.

Usage::

    python -m qfz.benchmarks.evaluate --results results

Reads every ``*.json`` record written by ``qfz.benchmarks.train`` /
``run_all`` and prints a markdown table comparing models per dataset.
"""

import argparse
import json
from pathlib import Path


def load_records(results_dir: str) -> list:
    """Load all benchmark records from a directory (excluding summary.json)."""
    records = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name == "summary.json":
            continue
        records.append(json.loads(path.read_text()))
    return records


def format_table(records: list) -> str:
    """Render benchmark records as a markdown table."""
    header = ("| dataset | model | test acc | trainable params | "
              "train time (s) | inference (ms/img) |")
    rule = "|---|---|---|---|---|---|"
    lines = [header, rule]
    for r in sorted(records, key=lambda r: (r["config"]["dataset"], r["config"]["model"])):
        c, m = r["config"], r["metrics"]
        lines.append(
            f"| {c['dataset']} | {c['model']} | {m['test_accuracy']:.4f} "
            f"| {m['parameters']['trainable']:,} | {m['train_time_s']:.1f} "
            f"| {m['inference']['ms_per_image']:.2f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    records = load_records(args.results)
    if not records:
        print(f"No benchmark records found in '{args.results}'.")
        return
    print(format_table(records))


if __name__ == "__main__":
    main()
