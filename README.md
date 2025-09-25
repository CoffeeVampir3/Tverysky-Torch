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
