"""Ablation grids over the swappable axes of the qfz filter layers.

Each grid varies exactly one axis of one filter layer, holding the rest
of the benchmark setup fixed (same subsets, epochs, head, seeds), so the
tables isolate the effect of that axis:

    entanglement  QPF CNOT pattern: horizontal / vertical / diagonal / ring
    encoding      Quanvolution data encoding: angle / basis
    circuit       Quanvolution circuit block: random / hardware_efficient / iqp
    trainable     Quanvolution random circuit, frozen vs trained angles
    mode          Quanvolution on RGB: per-channel (4 qubits) vs stacked
                  (12 qubits), CIFAR-10 and SVHN only

Usage::

    python -m qfz.benchmarks.ablations --grids entanglement encoding \
        --seeds 42 43 44 45 46 --out results/ablations

Writes one JSON per run plus ``summary.json`` / ``summary.md`` (mean
+/- std over seeds, one row per variant). Runs are resumable: existing
result files are skipped unless ``--overwrite``.
"""

import argparse
import json
import traceback
from pathlib import Path

from qfz.benchmarks.evaluate import aggregate_records, format_aggregate_table
from qfz.benchmarks.train import run_experiment

ALL_DATASETS = ["mnist", "fashionmnist", "cifar10", "svhn"]
RGB_DATASETS = ["cifar10", "svhn"]

# grid name -> list of (variant, model, datasets, layer_kwargs)
GRIDS = {
    "entanglement": [
        (f"qpf-{pattern}", "qpf", ALL_DATASETS, {"entanglement": pattern})
        for pattern in ("horizontal", "vertical", "diagonal", "ring")
    ],
    "encoding": [
        (f"quanv-{encoding}", "quanv", ALL_DATASETS, {"encoding": encoding})
        for encoding in ("angle", "basis")
    ],
    "circuit": [
        (f"quanv-{circuit}", "quanv", ALL_DATASETS, {"circuit": circuit})
        for circuit in ("random", "hardware_efficient", "iqp")
    ],
    "trainable": [
        ("quanv-frozen", "quanv", ALL_DATASETS, {"trainable": False}),
        ("quanv-trained", "quanv", ALL_DATASETS, {"trainable": True}),
    ],
    "mode": [
        ("quanv-perchannel", "quanv", RGB_DATASETS, {"per_channel": True}),
        # Stacked: all 3 channels of a 2x2 patch share one 12-qubit
        # circuit. Smaller patch batches keep the statevector memory low.
        ("quanv-stacked", "quanv", RGB_DATASETS,
         {"per_channel": False, "max_patch_batch": 1024}),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grids", nargs="+", default=list(GRIDS),
                        choices=list(GRIDS))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--root", default=None,
                        help="data root (default: $QFZ_DATA_ROOT or ./data)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/ablations")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for grid in args.grids:
        for variant, model, datasets, layer_kwargs in GRIDS[grid]:
            for dataset in datasets:
                for seed in args.seeds:
                    run_name = f"{dataset}_{variant}_seed{seed}"
                    result_path = out_dir / f"{run_name}.json"
                    if result_path.exists() and not args.overwrite:
                        print(f"[skip] {run_name} (exists)", flush=True)
                        records.append(json.loads(result_path.read_text()))
                        continue
                    print(f"[run ] {run_name}", flush=True)
                    try:
                        record = run_experiment(
                            dataset, model, epochs=args.epochs,
                            batch_size=args.batch_size,
                            train_size=args.train_size,
                            test_size=args.test_size, lr=args.lr, seed=seed,
                            root=args.root, device=args.device,
                            out_dir=args.out, progress=False,
                            layer_kwargs=layer_kwargs, variant=variant,
                            run_name=run_name)
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
