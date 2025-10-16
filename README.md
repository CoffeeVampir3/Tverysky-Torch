[![Discord](https://img.shields.io/discord/232596713892872193?logo=discord)](https://discord.gg/2JhHVh7CGu)

This has an implementation of the Tversky Layer that replaces linears as outlined by the excellent paper: https://arxiv.org/pdf/2506.11035

### ~

XOR test: https://github.com/CoffeeVampir3/Tverysky-Torch

Cifar10: https://github.com/CoffeeVampir3/Tversky-Cifar10/tree/main

Language model: https://github.com/CoffeeVampir3/Architecture-Tversky-All

### ~

XOR Test on 15 seeds 1000 epochs per seed:
```python
uv run python XORtest.py
```

This is using a variety of modifications that produced better empircal results at scale. (Asinh norm and some different initialization patterns)

Where the convergence is, out of the 15 random seeds, what percentage of networks arrived at 100% accuracy after 1000 epochs.
Sample results:
```
Features: 1 | Accuracy: 0.633±0.129 | Convergence: 0.000
Features: 2 | Accuracy: 0.867±0.160 | Convergence: 0.533
Features: 4 | Accuracy: 0.933±0.114 | Convergence: 0.733
Features: 8 | Accuracy: 0.933±0.114 | Convergence: 0.733
Features: 16 | Accuracy: 0.983±0.065 | Convergence: 0.933
Features: 32 | Accuracy: 0.983±0.065 | Convergence: 0.933
Features: 64 | Accuracy: 1.000±0.000 | Convergence: 1.000
Features: 128 | Accuracy: 0.983±0.065 | Convergence: 0.933
```
