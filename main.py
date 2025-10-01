import torch
import torch.nn as nn
import time
from modeling.TverskyLayer import TverskyLayer

def test_correctness():
    torch.manual_seed(42)
    batch_size, input_dim, num_prototypes, num_features = 10, 32*32, 5, 12
    layer = TverskyLayer(input_dim, num_prototypes, num_features, True)
    x = torch.randn(batch_size, input_dim)

    with torch.no_grad():
        result_regular = layer.forward(x)
        result_chunked = layer.forward_chunk(x, chunk_size=2)

    max_diff = torch.max(torch.abs(result_regular - result_chunked)).item()
    all_close = torch.allclose(result_regular, result_chunked, atol=1e-6)

    print(f"Max difference: {max_diff:.2e}")
    print(f"Results match: {all_close}")
    return all_close

def benchmark_speed():
    print("\nSpeed Benchmark")
    print("-" * 60)

    configs = [
        (32, 16, 10, 20),
        (128, 64, 50, 100),
        (512, 128, 100, 200),
    ]

    for batch_size, input_dim, num_prototypes, num_features in configs:
        torch.manual_seed(42)
        layer = TverskyLayer(input_dim, num_prototypes, num_features, True)
        x = torch.randn(batch_size, input_dim)

        if torch.cuda.is_available():
            layer = layer.cuda()
            x = x.cuda()
            for _ in range(5):
                _ = layer.forward(x)
            torch.cuda.synchronize()

        n_runs = 50 if batch_size <= 128 else 10

        start = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                output_regular = layer.forward(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_regular = (time.time() - start) / n_runs

        start = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                output_chunked = layer.forward_chunk(x, chunk_size=16)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_chunked = (time.time() - start) / n_runs

        print(f"\nBatch={batch_size}, Input={input_dim}, Proto={num_prototypes}, Feat={num_features}")
        print(f"  forward()        : {time_regular*1000:6.2f} ms")
        print(f"  forward_chunk()  : {time_chunked*1000:6.2f} ms")
        print(f"  Speedup          : {time_regular/time_chunked:6.2f}x")

def test_gradients():
    print("\nGradient Test")
    print("-" * 60)

    torch.manual_seed(42)
    input_dim, num_prototypes, num_features = 4, 3, 6
    layer1 = TverskyLayer(input_dim, num_prototypes, num_features, True)
    layer2 = TverskyLayer(input_dim, num_prototypes, num_features, True)
    layer2.load_state_dict(layer1.state_dict())

    x = torch.randn(5, input_dim, requires_grad=True)
    target = torch.randn(5, num_prototypes)

    out_regular = layer1.forward(x)
    out_chunked = layer2.forward_chunk(x, chunk_size=2)

    loss_regular = torch.mean((out_regular - target)**2)
    loss_chunked = torch.mean((out_chunked - target)**2)

    loss_regular.backward()
    loss_chunked.backward()

    for (name1, param1), (name2, param2) in zip(layer1.named_parameters(), layer2.named_parameters()):
        if param1.grad is not None and param2.grad is not None:
            diff = torch.max(torch.abs(param1.grad - param2.grad)).item()
            print(f"  {name1:12s}: {diff:.2e}")

def memory_test():
    print("\nMemory Test")
    print("-" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return

    batch_size, input_dim, num_prototypes, num_features = 256, 128, 100, 200
    layer = TverskyLayer(input_dim, num_prototypes, num_features, True).cuda()
    x = torch.randn(batch_size, input_dim).cuda()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output_regular = layer.forward(x)
    mem_regular = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output_chunked = layer.forward_chunk(x, chunk_size=16)
    mem_chunked = torch.cuda.max_memory_allocated()

    print(f"  forward()       : {mem_regular / 1024**2:6.1f} MB")
    print(f"  forward_chunk() : {mem_chunked / 1024**2:6.1f} MB")
    print(f"  Ratio           : {mem_chunked / mem_regular:6.2f}x")

def main():
    print("=" * 60)
    print("Tversky Layer Benchmark")
    print("=" * 60)

    print("\nCorrectness Test")
    print("-" * 60)
    test_correctness()

    benchmark_speed()
    test_gradients()
    memory_test()

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
