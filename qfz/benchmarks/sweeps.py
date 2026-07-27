"""Sensitivity sweeps over training budget: epochs and dataset size.

Two sweeps address the two standard objections to low-budget benchmarks:

    epochs   Are the 3-epoch comparisons an undertraining artifact? Sweep
             epochs at fixed data budget (grayscale datasets, where the
             fixed quantum filters beat the trained classical conv).
    budget   Is the fixed-filter advantage specific to the low-data
             regime? Sweep the training-set size at fixed epochs.

Usage::

    python -m qfz.benchmarks.sweeps --sweep epochs --out results/sweeps/epochs
    python -m qfz.benchmarks.sweeps --sweep budget --out results/sweeps/budget

Runs are tagged ``<model>-ep<epochs>`` / ``<model>-n<train_size>`` so the
standard aggregation groups them; existing result files are skipped
unless ``--overwrite``.
"""

import argparse
import json
import traceback
from pathlib import Path

from qfz.benchmarks.evaluate import aggregate_records, format_aggregate_table
from qfz.benchmarks.train import run_experiment

SWEEP_MODELS = ["classical", "randconv", "rff", "quanv", "qpf"]

SWEEPS = {
    # axis values, datasets, fixed kwargs
    "epochs": {
        "values": [3, 10, 30],
        "datasets": ["mnist", "fashionmnist"],
        "tag": "ep",
    },
    "budget": {
        "values": [500, 1000, 2000, 5000],
        "datasets": ["mnist", "fashionmnist", "cifar10", "svhn"],
        "tag": "n",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, choices=list(SWEEPS))
    parser.add_argument("--models", nargs="+", default=SWEEP_MODELS)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--root", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None,
                        help="default: results/sweeps/<sweep>")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    spec = SWEEPS[args.sweep]
    out_dir = Path(args.out) if args.out else Path("results/sweeps") / args.sweep
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for value in spec["values"]:
        for dataset in spec["datasets"]:
            for model in args.models:
                for seed in args.seeds:
                    variant = f"{model}-{spec['tag']}{value}"
                    run_name = f"{dataset}_{variant}_seed{seed}"
                    result_path = out_dir / f"{run_name}.json"
                    if result_path.exists() and not args.overwrite:
                        print(f"[skip] {run_name} (exists)", flush=True)
                        records.append(json.loads(result_path.read_text()))
                        continue
                    epochs = value if args.sweep == "epochs" else 3
                    train_size = 2000 if args.sweep == "epochs" else value
                    print(f"[run ] {run_name}", flush=True)
                    try:
                        record = run_experiment(
                            dataset, model, epochs=epochs,
                            batch_size=args.batch_size, train_size=train_size,
                            test_size=args.test_size, lr=args.lr, seed=seed,
                            root=args.root, device=args.device,
                            out_dir=str(out_dir), progress=False,
                            variant=variant, run_name=run_name)
                        records.append(record)
                        m = record["metrics"]
                        print(f"       acc={m['test_accuracy']:.4f} "
                              f"train={m['train_time_s']:.1f}s", flush=True)
                    except Exception:
                        print(f"[fail] {run_name}\n{traceback.format_exc()}",
                              flush=True)

    (out_dir / "summary.json").write_text(json.dumps(records, indent=2))
    table = format_aggregate_table(aggregate_records(records))
    (out_dir / "summary.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
