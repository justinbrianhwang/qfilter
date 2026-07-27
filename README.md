# Quantum Filter Zoo (qfz)

<img width="1376" height="768" alt="A-clean,-modern-technical-illustration-for-a-machine-learning-open-sour…-image-1" src="https://github.com/user-attachments/assets/a17375b9-c4d4-4860-8b28-78a118b8cba1" />


A PyTorch library of **quantum filter layers** that plug into classical deep
learning models. Built on [PennyLane](https://pennylane.ai/).

> **Disclaimer**: This library is for research and educational experiments,
> not a proven quantum advantage framework. Results on small benchmarks do
> not imply quantum advantage.

## Features

- **`Quanvolution2D`** — quantum convolutional filter (Henderson et al., 2020).
  Full `nn.Conv2d`-style geometry: `kernel_size`, `stride`, `padding`,
  `dilation` (int or tuple).
- **`QPF`** — fixed, parameter-free 4-qubit quantum preprocessing filter
  (Riaz et al., 2023) with selectable CNOT entanglement patterns
  (`horizontal` / `vertical` / `diagonal` / `ring`).
- **`PQCConv2D`** — trainable parameterized quantum convolution
  (hardware-efficient ansatz, trained by backprop through the simulator).
- Encodings: angle (`RY(πx)`), basis (thresholded bit-flip).
- Circuits: `random` (seeded structure), `hardware_efficient`, `iqp` —
  all registered and swappable via the `circuit=` argument.
- **Per-channel mode** for RGB images: the same 4-qubit filter is applied to
  each channel independently (depthwise-style), keeping simulation cheap.
- **Dataset registry**: MNIST, FashionMNIST, CIFAR-10, SVHN built in; any
  torch dataset can be registered and used with the same benchmark tooling.
- **Multi-seed benchmark grid** (`run_all`) reporting accuracy (mean ± std
  over seeds), parameter count, training time, and inference time against
  three architecture-matched classical baselines: a *trained* CNN, a
  *frozen random* CNN, and a *frozen spectrum-matched random Fourier
  feature* filter (`RFFFilter2D`) — the controls the dequantization
  literature prescribes for fixed quantum filters.
- **Ablation grids** (`ablations`) over every swappable axis: entanglement
  pattern, encoding, circuit block, trainable vs. frozen angles, and
  per-channel vs. stacked multi-channel mode — plus **budget sweeps**
  (`sweeps`) over epochs and training-set size.

## Installation

```bash
git clone https://github.com/justinbrianhwang/qfilter.git
cd qfilter

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

# full grid: 4 datasets x {classical, randconv, rff, quanv, qpf, pqc} x 10 seeds
python -m qfz.benchmarks.run_all --datasets mnist fashionmnist cifar10 svhn \
    --seeds 42 43 44 45 46 47 48 49 50 51 --out results/multiseed

# ablations over entanglement / encoding / circuit / trainability / mode
python -m qfz.benchmarks.ablations --out results/ablations

# sensitivity sweeps over epochs and training-set size
python -m qfz.benchmarks.sweeps --sweep epochs
python -m qfz.benchmarks.sweeps --sweep budget

# print the aggregated comparison table (mean ± std over seeds)
python -m qfz.benchmarks.evaluate --results results/multiseed

# reproduce every reported statistic (Welch/paired t, Holm, Cohen's d, CIs)
python scripts/stats_analysis.py
```

Each run saves a self-describing JSON (config + metrics + library
versions + code revision); runs are resumable and aggregated per
(dataset, model) across seeds. Three classical baselines mirror the
hybrid architecture exactly (same patch geometry + identical linear
head): `classical` (trained conv + Tanh), `randconv` (the same conv
*frozen* at random initialization), and `rff` (frozen random Fourier
features drawn from the quantum filters' own frequency spectrum
{-1,0,1}^k² — the dequantization-aware control).

### Results

2,000 training / 1,000 test images per dataset, 3 epochs, batch 64, Adam
1e-3, **10 seeds (mean ± std)**, CPU (`default.qubit` simulator). Quantum
filters use 4 qubits (per-channel mode on RGB). These are small-subset,
low-data-regime benchmarks, **not** evidence of quantum advantage.

| dataset | classical | randconv (frozen) | rff (frozen) | quanv | qpf | pqc |
|---|---|---|---|---|---|---|
| mnist | .803 ± .019 | .763 ± .028 | .788 ± .037 | **.827 ± .019** | .816 ± .020 | .736 ± .050 |
| fashionmnist | .726 ± .014 | .706 ± .020 | .739 ± .035 | .747 ± .015 | **.767 ± .015** | .718 ± .023 |
| cifar10 | .327 ± .015 | .298 ± .026 | .312 ± .034 | **.336 ± .027** | .335 ± .018 | .324 ± .026 |
| svhn | .188 ± .019 | .173 ± .025 | .195 ± .029 | .198 ± .014 | .204 ± .026 | **.208 ± .017** |

Observations on these subsets (Welch t-tests over 10 seeds, Holm-corrected
per model across the four datasets; paired tests agree):

- The fixed quantum filters (`quanv`, `qpf`) outperform the **frozen random
  conv** of identical geometry on all four datasets (Holm-corrected
  p ≤ 0.017, Cohen's d 1.2-3.5) — but against the **spectrum-matched RFF
  control** the point estimates stay positive while most comparisons lose
  significance. The naive frozen control overstates how distinctive the
  quantum feature maps are; the dequantization-aware control absorbs most
  of the gap.
- Against the *trained* classical conv they win on the grayscale datasets
  (e.g. qpf on fashionmnist: +4.1 pts, Holm p < 0.001) and are on par on RGB.
- Training the circuit angles moves accuracy by at most +0.4 pts
  (paired 95% CIs; nothing detectable on RGB), and `pqc` never
  significantly exceeds the fixed filters.
- Inference under simulation remains 10-100x slower than the classical conv.
- Budget sweeps (epochs 3→30, train size 500→5,000; `results/sweeps/`) show
  the edge over the *trained* conv is a low-budget effect — it disappears
  once the classical model is given 30 epochs or 5,000 images — while the
  ordering above the frozen controls is stable across both axes.

Raw records: [results/multiseed/](results/multiseed/) and
[results/ablations/](results/ablations/), chart:
`results/multiseed/summary.png`. Ablation grids (entanglement patterns,
encodings, circuit blocks, trainability, per-channel vs. stacked mode) are
summarized in [results/ablations/summary.md](results/ablations/summary.md).

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
├── benchmarks/  # train / evaluate / metrics / run_all / ablations / sweeps
└── utils/       # seeding, visualization
```

## Reproducibility

Circuit structures and initial angles are fixed by each layer's `seed`
argument; `qfz.utils.set_seed(42)` fixes the remaining randomness (weight
init, shuffling, subsampling). The exact environment used for the
published benchmarks is pinned in `requirements-lock.txt`
(Python 3.11.15); the test suite runs on every push via GitHub Actions.

## References

- Henderson et al., *Quanvolutional Neural Networks: Powering Image
  Recognition with Quantum Circuits*, Quantum Machine Intelligence (2020).
- Riaz et al., *Development of a Novel Quantum Pre-processing Filter to
  Improve Image Classification Accuracy of Neural Network Models*,
  [arXiv:2308.11112](https://arxiv.org/abs/2308.11112) (2023) — the QPF
  design that `QPF` implements and generalizes.
- Riaz et al., *Application of Quantum Pre-Processing Filter for Binary
  Image Classification with Small Samples*,
  [arXiv:2308.14930](https://arxiv.org/abs/2308.14930) (2023).
- Schuld et al., *Effect of Data Encoding on the Expressive Power of
  Variational Quantum Machine Learning Models*,
  [arXiv:2008.08605](https://arxiv.org/abs/2008.08605) (2021), and
  Landman et al., *Classically Approximating Variational Quantum Machine
  Learning with Random Fourier Features*,
  [arXiv:2210.13200](https://arxiv.org/abs/2210.13200) (ICLR 2023) — the
  basis for the spectrum-matched `RFFFilter2D` control.
- PennyLane demo: [Quanvolutional Neural Networks](https://pennylane.ai/qml/demos/tutorial_quanvolution).

## License

MIT
