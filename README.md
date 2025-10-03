[![Discord](https://img.shields.io/discord/232596713892872193?logo=discord)](https://discord.gg/2JhHVh7CGu)

This has an implementation of the Tversky Layer that replaces linears as outlined by the excellent paper: https://arxiv.org/pdf/2506.11035


XOR Test on 15 seeds 1000 epochs per seed:
```python
uv run python XORtest.py
```

Under initialization conditions (Biasing for larger network sizes):
```python
approximate_sharpness=13,
prototypes -> uniform(x, -0.15, 0.15),
features -> uniform(x, -0.2, 0.2),
alpha -> uniform(0.004, 0.25),
beta -> uniform(0.001, 0.004),
theta -> uniform(0.07, 0.13)
```

Where the convergence is, out of the 15 random seeds, what percentage of networks arrived at 100% accuracy.
Sample results:
```
Features: 1 | Accuracy: 0.600±0.228 | Convergence: 0.000
Features: 2 | Accuracy: 0.683±0.148 | Convergence: 0.000
Features: 4 | Accuracy: 0.783±0.088 | Convergence: 0.133
Features: 8 | Accuracy: 0.900±0.127 | Convergence: 0.600
Features: 16 | Accuracy: 0.967±0.088 | Convergence: 0.867
Features: 32 | Accuracy: 0.983±0.065 | Convergence: 0.933
Features: 64 | Accuracy: 1.000±0.000 | Convergence: 1.000
Features: 128 | Accuracy: 1.000±0.000 | Convergence: 1.000
```
