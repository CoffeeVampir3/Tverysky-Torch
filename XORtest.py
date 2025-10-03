import torch
import torch.nn as nn
import torch.optim as optim
import time
from modeling.TverskyLayer import TverskyLayer

def test_tversky_xor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=torch.float32).to(device)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long).to(device)

    input_dim = 2
    num_prototypes = 2
    feature_counts = [1, 2, 4, 8, 16, 32]
    results = []

    for num_features in feature_counts:
        seed_results = []
        for seed in range(15):
            torch.manual_seed(seed)
            model = TverskyLayer(
                input_dim,
                num_prototypes,
                num_features,
                prototype_init=lambda x: torch.nn.init.uniform_(x, -0.15, 0.15),
                feature_init=lambda x: torch.nn.init.uniform_(x, -0.10, 0.10),).to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            model.train()
            start_time = time.time()
            for epoch in range(1000):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            elapsed_time = time.time() - start_time

            model.eval()
            with torch.no_grad():
                final_outputs = model(X)
                _, predicted = torch.max(final_outputs, 1)
                accuracy = (predicted == y).float().mean().item()
                converged = accuracy == 1.0

            seed_results.append({
                'accuracy': accuracy,
                'converged': converged
            })

        accuracies = torch.tensor([r['accuracy'] for r in seed_results])
        convergence_rate = sum(1 for r in seed_results if r['converged']) / len(seed_results)
        mean_accuracy = accuracies.mean().item()
        std_accuracy = accuracies.std().item()

        results.append({
            'num_features': num_features,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'convergence_rate': convergence_rate
        })
        print(f"Features: {num_features} | Accuracy: {mean_accuracy:.3f}±{std_accuracy:.3f} | Convergence: {convergence_rate:.3f}")

    return results

if __name__ == "__main__":
    results = test_tversky_xor()
