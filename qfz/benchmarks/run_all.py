"""Run the full benchmark grid: every dataset x every model.

Usage::

    python -m qfz.benchmarks.run_all --datasets mnist fashionmnist cifar10 svhn \
        --models classical quanv qpf pqc --epochs 3 --train-size 2000 --test-size 1000

Writes one JSON per run plus ``summary.json`` and ``summary.md`` to the
output directory, printing the comparison table at the end. Runs are
resumable: existing result files are skipped unless ``--overwrite``.
"""

import argparse
import json
import traceback
from pathlib import Path

from qfz.benchmarks.evaluate import format_table
from qfz.benchmarks.train import MODELS, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        default=["mnist", "fashionmnist", "cifar10", "svhn"])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", default=None,
                        help="data root (default: $QFZ_DATA_ROOT or ./data)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for dataset in args.datasets:
        for model in args.models:
            result_path = out_dir / f"{dataset}_{model}.json"
            if result_path.exists() and not args.overwrite:
                print(f"[skip] {dataset}/{model} (exists)", flush=True)
                records.append(json.loads(result_path.read_text()))
                continue
            print(f"[run ] {dataset}/{model}", flush=True)
            try:
                record = run_experiment(
                    dataset, model, epochs=args.epochs, batch_size=args.batch_size,
                    train_size=args.train_size, test_size=args.test_size,
                    lr=args.lr, seed=args.seed, root=args.root,
                    device=args.device, out_dir=args.out, progress=False)
                records.append(record)
                m = record["metrics"]
                print(f"       acc={m['test_accuracy']:.4f} "
                      f"train={m['train_time_s']:.1f}s", flush=True)
            except Exception:
                print(f"[fail] {dataset}/{model}\n{traceback.format_exc()}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(records, indent=2))
    table = format_table(records)
    (out_dir / "summary.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
