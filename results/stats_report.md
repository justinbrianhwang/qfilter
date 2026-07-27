# Statistical reanalysis report

## quanv vs randconv  (Holm family = 4 datasets)
| dataset | diff | Welch p | Welch p (Holm) | paired p | paired p (Holm) | Cohen's d | n |
|---|---|---|---|---|---|---|---|
| mnist | +0.0640 | 2e-05 | 9e-05 | 9e-05 | 0.00035 | 2.66 | 10 |
| fashionmnist | +0.0409 | 6e-05 | 0.00018 | 0.0001 | 0.00035 | 2.38 | 10 |
| cifar10 | +0.0380 | 0.0051 | 0.01 | 0.0072 | 0.014 | 1.43 | 10 |
| svhn | +0.0250 | 0.017 | 0.017 | 0.033 | 0.033 | 1.21 | 10 |

## qpf vs randconv  (Holm family = 4 datasets)
| dataset | diff | Welch p | Welch p (Holm) | paired p | paired p (Holm) | Cohen's d | n |
|---|---|---|---|---|---|---|---|
| mnist | +0.0529 | 0.00018 | 0.00055 | 7e-05 | 0.00021 | 2.15 | 10 |
| fashionmnist | +0.0606 | 5e-07 | 2e-06 | 9e-06 | 4e-05 | 3.53 | 10 |
| cifar10 | +0.0368 | 0.0021 | 0.0042 | 0.00041 | 0.00082 | 1.64 | 10 |
| svhn | +0.0304 | 0.017 | 0.017 | 0.0045 | 0.0045 | 1.18 | 10 |

## rff vs randconv  (Holm family = 4 datasets)
| dataset | diff | Welch p | Welch p (Holm) | paired p | paired p (Holm) | Cohen's d | n |
|---|---|---|---|---|---|---|---|
| mnist | +0.0252 | 0.1 | 0.28 | 0.1 | 0.2 | 0.77 | 10 |
| fashionmnist | +0.0322 | 0.024 | 0.094 | 0.022 | 0.088 | 1.13 | 10 |
| cifar10 | +0.0134 | 0.34 | 0.34 | 0.37 | 0.37 | 0.44 | 10 |
| svhn | +0.0216 | 0.092 | 0.28 | 0.067 | 0.2 | 0.80 | 10 |

## quanv vs rff  (Holm family = 4 datasets)
| dataset | diff | Welch p | Welch p (Holm) | paired p | paired p (Holm) | Cohen's d | n |
|---|---|---|---|---|---|---|---|
| mnist | +0.0388 | 0.01 | 0.042 | 0.024 | 0.071 | 1.33 | 10 |
| fashionmnist | +0.0087 | 0.48 | 0.97 | 0.5 | 1 | 0.32 | 10 |
| cifar10 | +0.0246 | 0.092 | 0.28 | 0.0017 | 0.0066 | 0.80 | 10 |
| svhn | +0.0034 | 0.74 | 0.97 | 0.77 | 1 | 0.15 | 10 |

## qpf vs rff  (Holm family = 4 datasets)
| dataset | diff | Welch p | Welch p (Holm) | paired p | paired p (Holm) | Cohen's d | n |
|---|---|---|---|---|---|---|---|
| mnist | +0.0277 | 0.055 | 0.17 | 0.052 | 0.15 | 0.94 | 10 |
| fashionmnist | +0.0284 | 0.036 | 0.14 | 0.024 | 0.097 | 1.06 | 10 |
| cifar10 | +0.0234 | 0.076 | 0.17 | 0.049 | 0.15 | 0.86 | 10 |
| svhn | +0.0088 | 0.49 | 0.49 | 0.42 | 0.42 | 0.32 | 10 |

## quanv vs classical  (Holm family = 4 datasets)
| dataset | diff | Welch p | Welch p (Holm) | paired p | paired p (Holm) | Cohen's d | n |
|---|---|---|---|---|---|---|---|
| mnist | +0.0235 | 0.012 | 0.035 | 0.014 | 0.043 | 1.26 | 10 |
| fashionmnist | +0.0211 | 0.0038 | 0.015 | 0.011 | 0.043 | 1.49 | 10 |
| cifar10 | +0.0088 | 0.38 | 0.38 | 0.42 | 0.46 | 0.41 | 10 |
| svhn | +0.0105 | 0.18 | 0.36 | 0.23 | 0.46 | 0.63 | 10 |

## qpf vs classical  (Holm family = 4 datasets)
| dataset | diff | Welch p | Welch p (Holm) | paired p | paired p (Holm) | Cohen's d | n |
|---|---|---|---|---|---|---|---|
| mnist | +0.0124 | 0.17 | 0.41 | 0.078 | 0.23 | 0.64 | 10 |
| fashionmnist | +0.0408 | 5e-06 | 2e-05 | 2e-05 | 7e-05 | 2.89 | 10 |
| cifar10 | +0.0076 | 0.31 | 0.41 | 0.21 | 0.27 | 0.47 | 10 |
| svhn | +0.0159 | 0.14 | 0.41 | 0.13 | 0.27 | 0.70 | 10 |

## pqc vs classical  (Holm family = 4 datasets)
| dataset | diff | Welch p | Welch p (Holm) | paired p | paired p (Holm) | Cohen's d | n |
|---|---|---|---|---|---|---|---|
| mnist | -0.0675 | 0.002 | 0.0078 | 0.006 | 0.024 | -1.79 | 10 |
| fashionmnist | -0.0083 | 0.34 | 0.69 | 0.34 | 0.68 | -0.44 | 10 |
| cifar10 | -0.0029 | 0.76 | 0.76 | 0.74 | 0.74 | -0.14 | 10 |
| svhn | +0.0201 | 0.022 | 0.067 | 0.01 | 0.03 | 1.12 | 10 |

## pqc vs randconv  (Holm family = 4 datasets)
| dataset | diff | Welch p | Welch p (Holm) | paired p | paired p (Holm) | Cohen's d | n |
|---|---|---|---|---|---|---|---|
| mnist | -0.0270 | 0.16 | 0.32 | 0.23 | 0.46 | -0.66 | 10 |
| fashionmnist | +0.0115 | 0.24 | 0.32 | 0.26 | 0.46 | 0.54 | 10 |
| cifar10 | +0.0263 | 0.036 | 0.11 | 0.022 | 0.065 | 1.01 | 10 |
| svhn | +0.0346 | 0.0025 | 0.01 | 0.0061 | 0.024 | 1.60 | 10 |

## quanv trained vs frozen (ablation, paired 95% CI of difference)
| dataset | mean diff | 95% CI | paired p | n |
|---|---|---|---|---|
| mnist | +0.0036 | [+0.0007, +0.0065] | 0.025 | 5 |
| fashionmnist | +0.0038 | [+0.0016, +0.0060] | 0.009 | 5 |
| cifar10 | +0.0022 | [-0.0017, +0.0061] | 0.19 | 5 |
| svhn | -0.0008 | [-0.0052, +0.0036] | 0.64 | 5 |

## Exact quoted figures
- pqc vs classical (MNIST): diff = -0.0675 (-6.7 pts), std ratio = 2.7x, variance ratio = 7.4x
