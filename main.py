import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from modeling.TverskyLayer import TverskyLayer

def test_correctness():
    torch.manual_seed(42)
    batch_size, input_dim, num_prototypes, num_features = 10, 32*32, 5, 12
    layer = TverskyLayer(input_dim, num_prototypes, num_features)
    x = torch.randn(batch_size, input_dim)

    with torch.no_grad():
        result_slow = layer.forward_bad_slow(x)
        result_fast = layer.forward(x)

    max_diff = torch.max(torch.abs(result_slow - result_fast)).item()
    all_close = torch.allclose(result_slow, result_fast, atol=1e-6)

    print(f"Max diff: {max_diff:.2e}, Identical: {all_close}")
    return all_close

def benchmark_speed():
    configs = [
        (32, 16, 10, 20),
        (128, 64, 50, 100),
        (512, 128, 100, 200),
    ]

    for batch_size, input_dim, num_prototypes, num_features in configs:
        torch.manual_seed(42)
        layer = TverskyLayer(input_dim, num_prototypes, num_features)
        x = torch.randn(batch_size, input_dim)

        if torch.cuda.is_available():
            layer = layer.cuda()
            x = x.cuda()
            for _ in range(5):
                _ = layer.forward(x)
            torch.cuda.synchronize()

        n_runs = 50 if batch_size <= 128 else 10

        start_time = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = layer.forward_bad_slow(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        slow_time = (time.time() - start_time) / n_runs

        start_time = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = layer.forward(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fast_time = (time.time() - start_time) / n_runs

        speedup = slow_time / fast_time if fast_time > 0 else float('inf')
        print(f"Config {batch_size}x{input_dim}: {slow_time*1000:.2f}ms -> {fast_time*1000:.2f}ms ({speedup:.1f}x)")

def test_gradients():
    torch.manual_seed(42)
    input_dim, num_prototypes, num_features = 4, 3, 6
    layer1 = TverskyLayer(input_dim, num_prototypes, num_features)
    layer2 = TverskyLayer(input_dim, num_prototypes, num_features)
    layer2.load_state_dict(layer1.state_dict())

    x = torch.randn(5, input_dim, requires_grad=True)
    target = torch.randn(5, num_prototypes)

    out1 = layer1.forward_bad_slow(x)
    out2 = layer2.forward(x)
    loss1 = torch.mean((out1 - target)**2)
    loss2 = torch.mean((out2 - target)**2)
    loss1.backward()
    loss2.backward()

    grad_diffs = []
    for (name1, param1), (name2, param2) in zip(layer1.named_parameters(), layer2.named_parameters()):
        if param1.grad is not None and param2.grad is not None:
            diff = torch.max(torch.abs(param1.grad - param2.grad)).item()
            grad_diffs.append(diff)
            print(f"{name1}: {diff:.2e}")

    max_grad_diff = max(grad_diffs) if grad_diffs else 0
    print(f"Max grad diff: {max_grad_diff:.2e}")
    return max_grad_diff < 1e-6

def memory_test():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    torch.cuda.empty_cache()
    batch_size, input_dim, num_prototypes, num_features = 256, 128, 100, 200
    layer = TverskyLayer(input_dim, num_prototypes, num_features).cuda()
    x = torch.randn(batch_size, input_dim).cuda()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = layer.forward_bad_slow(x)
    slow_memory = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = layer.forward(x)
    fast_memory = torch.cuda.max_memory_allocated()

    print(f"Memory: {slow_memory / 1024**2:.1f}MB -> {fast_memory / 1024**2:.1f}MB (ratio: {fast_memory / slow_memory:.2f})")

def main():
    correctness_ok = test_correctness()
    if correctness_ok:
        benchmark_speed()
        test_gradients()
        memory_test()

if __name__ == "__main__":
    main()
