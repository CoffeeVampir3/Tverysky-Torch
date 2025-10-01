[![Discord](https://img.shields.io/discord/232596713892872193?logo=discord)](https://discord.gg/2JhHVh7CGu)

This has an implementation of the Tversky Layer that replaces linears as outlined by the excellent paper: https://arxiv.org/pdf/2506.11035

Included in the TveryskyLayer.py is two versions, a `forward_bad_slow` and accompanying `tversky_similarity` which should only be used as conceptual examples. Use the regular `forward` which has been vectorized in practice.

The only notable thing is in `modeling/TverskyLayer.py` and it's pretty simple.

To run the vectorized difference test in main.py, simply

```python
uv run python main.py
```

XOR Test similar to the paper:
```python
uv run python XORtest.py
```
Sample results:
```
Features: 1 | Accuracy: 0.800±0.274 | Convergence: 0.600
Features: 2 | Accuracy: 0.850±0.137 | Convergence: 0.400
Features: 4 | Accuracy: 0.950±0.112 | Convergence: 0.800
Features: 8 | Accuracy: 0.900±0.137 | Convergence: 0.600
Features: 16 | Accuracy: 0.900±0.137 | Convergence: 0.600
Features: 32 | Accuracy: 0.950±0.112 | Convergence: 0.800
```
