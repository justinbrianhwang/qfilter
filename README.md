# Quantum Filter Zoo (qfz)

A PyTorch library of **quantum filter layers** that plug into classical deep
learning models. Built on [PennyLane](https://pennylane.ai/).

> **Disclaimer**: This library is for research and educational experiments,
> not a proven quantum advantage framework. Results on small benchmarks do
> not imply quantum advantage.

## Features

- **`Quanvolution2D`** — quantum convolutional filter (Henderson et al., 2020).
  Full `nn.Conv2d`-style geometry: `kernel_size`, `stride`, `padding`,
  `dilation` (int or tuple).
- **`QPF`** — fixed, parameter-free 4-qubit quantum preprocessing filter with
  selectable CNOT entanglement patterns (`horizontal` / `vertical` /
  `diagonal` / `ring`).
- **`PQCConv2D`** — trainable parameterized quantum convolution
  (hardware-efficient ansatz, trained by backprop through the simulator).
- Encodings: angle (`RY(πx)`), basis (thresholded bit-flip).
- Circuits: `random` (seeded structure), `hardware_efficient`, `iqp` —
  all registered and swappable via the `circuit=` argument.
- **Per-channel mode** for RGB images: the same 4-qubit filter is applied to
  each channel independently (depthwise-style), keeping simulation cheap.
- **Dataset registry**: MNIST, FashionMNIST, CIFAR-10, SVHN built in; any
  torch dataset can be registered and used with the same benchmark tooling.
- Benchmark grid (`run_all`) reporting accuracy, parameter count,
  training time, and inference time against an architecture-matched
  classical CNN baseline.

## Installation

```bash
git clone https://github.com/<you>/quantum-filter-zoo.git
cd quantum-filter-zoo

# (recommended) create a conda environment
conda env create -f environment.yml
conda activate qfz

pip install -e .
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, PennyLane ≥ 0.35. CPU is enough for
the small experiments below.

## Quick start

```python
import torch
from qfz.layers import Quanvolution2D

model = torch.nn.Sequential(
    Quanvolution2D(
        in_channels=1,
        out_channels=4,
        kernel_size=2,       # patch_size=2 also accepted
        stride=2,            # defaults to kernel_size
        padding=0,
        encoding="angle",
        circuit="random",
        backend="pennylane",
    ),
    torch.nn.Flatten(),
    torch.nn.Linear(4 * 14 * 14, 10),
)

x = torch.rand(8, 1, 28, 28)   # values must be in [0, 1]
logits = model(x)              # [8, 10]
```

All layers are standard `nn.Module`s and work with any input size, channel
count, and downstream architecture. By default the quantum filter is
**non-trainable** (a fixed random feature extractor, as in the original
quanvolution paper); pass `trainable=True` to learn the circuit angles.

More layers:

```python
from qfz.layers import QPF, PQCConv2D

qpf = QPF(in_channels=3, entanglement="ring")          # parameter-free, RGB -> 12 maps
pqc = PQCConv2D(in_channels=3, out_channels=12,        # trainable circuit angles
                per_channel=True, n_layers=2)
```

## How it works

`Quanvolution2D` slides a `kernel_size` window over the image. Each patch is
mapped onto qubits via angle encoding (`RY(π·x)`), passed through a quantum
circuit, and the Pauli-Z expectation value of qubit *k* becomes output
channel *k*. Outputs lie in `[-1, 1]`.

```
[B, C, H, W] ──unfold──▶ patches ──encode──▶ |ψ⟩ ──circuit──▶ ⟨Z_k⟩ ──fold──▶ [B, K, H', W']
```

For multi-channel inputs choose between **stacked** mode (all channels of a
patch share one circuit, `C·k²` qubits) and **per-channel** mode
(`per_channel=True`, one `k²`-qubit circuit applied to every channel —
recommended for RGB).

## Datasets

```python
from qfz.datasets import get_dataloaders, register_dataset, DatasetInfo

train, test, info = get_dataloaders("cifar10", batch_size=64, train_size=2000)

@register_dataset("mydata")            # plug in your own dataset
def _build(root):
    ...
    return train_ds, test_ds, DatasetInfo("mydata", in_channels=3,
                                          num_classes=5, img_size=(64, 64))
```

Images must be scaled to `[0, 1]` (the quantum encodings assume it).

Data is stored under `$QFZ_DATA_ROOT` if that environment variable is set
(useful for sharing one dataset directory across projects), otherwise under
`./data`. Any `root=` argument overrides both.

## Benchmarks

```bash
# single run
python -m qfz.benchmarks.train --dataset mnist --model quanv --epochs 3

# full grid: 4 datasets x {classical, quanv, qpf, pqc}
python -m qfz.benchmarks.run_all --datasets mnist fashionmnist cifar10 svhn

# print the comparison table from saved results
python -m qfz.benchmarks.evaluate --results results
```

Each run saves a self-describing JSON (config + metrics) under `results/`.
The classical baseline mirrors the hybrid architecture exactly (same
geometry conv + Tanh + identical linear head), so differences isolate the
effect of the quantum filter.

### Results

2,000 training / 1,000 test images per dataset, 3 epochs, batch 64, Adam
1e-3, seed 42, CPU (`default.qubit` simulator). Quantum filters use 4 qubits
(per-channel mode on RGB). These are small-subset sanity benchmarks, **not**
evidence of quantum advantage.

| dataset | model | test acc | trainable params | train time (s) | inference (ms/img) |
|---|---|---|---|---|---|
| mnist | classical | 0.807 | 7,870 | 0.6 | <0.01 |
| mnist | quanv | 0.783 | 7,850 | 4.5 | 0.65 |
| mnist | qpf | **0.829** | 7,850 | 3.2 | 0.36 |
| mnist | pqc | 0.787 | 7,866 | 17.3 | 1.17 |
| fashionmnist | classical | 0.755 | 7,870 | 0.5 | <0.01 |
| fashionmnist | quanv | 0.758 | 7,850 | 4.7 | 0.68 |
| fashionmnist | qpf | **0.790** | 7,850 | 3.2 | 0.44 |
| fashionmnist | pqc | 0.739 | 7,866 | 17.4 | 1.17 |
| cifar10 | classical | 0.320 | 30,886 | 0.6 | <0.01 |
| cifar10 | quanv | **0.379** | 30,730 | 16.5 | 2.62 |
| cifar10 | qpf | 0.367 | 30,730 | 10.3 | 1.60 |
| cifar10 | pqc | 0.371 | 30,746 | 66.8 | 4.54 |
| svhn | classical | 0.162 | 30,886 | 0.6 | <0.01 |
| svhn | quanv | 0.193 | 30,730 | 15.0 | 2.37 |
| svhn | qpf | **0.208** | 30,730 | 9.9 | 1.50 |
| svhn | pqc | 0.192 | 30,746 | 64.6 | 4.54 |

The quantum filters are competitive with (and on these subsets often above)
the matched classical conv at equal head capacity, at the cost of 10-100x
slower inference under simulation. Raw records: [results/](results/),
chart: `results/summary.png`.

## Examples

```bash
python examples/mnist_quanvolution.py        # fixed random quanvolution
python examples/fashionmnist_qpf.py ring     # parameter-free QPF, 4 patterns
python examples/cifar10_pqc_conv.py          # trainable PQC on RGB
```

## Project structure

```
qfz/
├── layers/      # QuantumFilter2d base, Quanvolution2D, QPF, PQCConv2D, encodings
├── circuits/    # random, hardware-efficient, IQP circuit blocks (registry)
├── models/      # HybridCNN wrapper + matched ClassicalCNN baseline
├── datasets/    # dataset registry: mnist/fashionmnist/cifar10/svhn + custom
├── benchmarks/  # train / evaluate / metrics / run_all
└── utils/       # seeding, visualization
```

## Reproducibility

Circuit structures and initial angles are fixed by each layer's `seed`
argument; `qfz.utils.set_seed(42)` fixes the remaining randomness (weight
init, shuffling, subsampling).

## References

- Henderson et al., *Quanvolutional Neural Networks: Powering Image
  Recognition with Quantum Circuits*, Quantum Machine Intelligence (2020).
- PennyLane demo: [Quanvolutional Neural Networks](https://pennylane.ai/qml/demos/tutorial_quanvolution).

## License

MIT
