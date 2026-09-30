"""
CSF407 Artificial Intelligence
Lab: Neural Models - Learning, Depth, Activations, and Output Layers

Run:
    python lab_neural_models.py

The script implements:
1. XOR problem specification and linear baseline
2. 2-2-1 binary XOR model with tanh hidden activation
3. Backpropagation/gradient inspection
4. Zero-initialization symmetry experiment
5. Sigmoid/tanh/ReLU activation experiment
6. Three-class sensor classification with softmax + cross-entropy
"""

import torch
import numpy as np

torch.set_printoptions(precision=6, sci_mode=False)

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
Y_BINARY = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
Y_THREE = torch.tensor([0, 1, 1, 2], dtype=torch.long)


def activation(name, x):
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "relu":
        return torch.relu(x)
    raise ValueError(f"Unknown activation: {name}")


class BinaryXORNet(torch.nn.Module):
    def __init__(self, hidden_activation="tanh"):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 1)
        self.hidden_activation = hidden_activation

    def forward(self, x):
        h = activation(self.hidden_activation, self.fc1(x))
        return self.fc2(h)  # logits; sigmoid is applied only for probabilities


class ThreeClassNet(torch.nn.Module):
    def __init__(self, hidden_activation="tanh"):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 3)
        self.hidden_activation = hidden_activation

    def forward(self, x):
        h = activation(self.hidden_activation, self.fc1(x))
        return self.fc2(h)  # three logits per example


def train_binary(hidden_activation="tanh", seed=42, steps=5000, lr=1.0):
    torch.manual_seed(seed)
    model = BinaryXORNet(hidden_activation)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        initial_loss = loss_fn(model(X), Y_BINARY).item()

    # First backward pass: used for the requested gradient inspection.
    optimizer.zero_grad()
    loss = loss_fn(model(X), Y_BINARY)
    loss.backward()
    first_layer_gradient = model.fc1.weight.grad.detach().clone()
    early_gradient_norm = torch.linalg.vector_norm(first_layer_gradient).item()
    optimizer.step()

    # Remaining training steps.
    for _ in range(steps - 1):
        optimizer.zero_grad()
        loss = loss_fn(model(X), Y_BINARY)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits).squeeze(1)
        predictions = (probabilities >= 0.5).long()
        correct = int((predictions == Y_BINARY.squeeze(1).long()).sum())

    return {
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": loss.item(),
        "probabilities": probabilities,
        "predictions": predictions,
        "correct": correct,
        "gradient": first_layer_gradient,
        "gradient_norm": early_gradient_norm,
    }


def symmetry_experiment(hidden_activation="tanh", steps=5, lr=1.0):
    """All parameters are set to zero so the two hidden units start identically."""
    torch.manual_seed(42)
    model = BinaryXORNet(hidden_activation)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rows = []

    for step in range(steps + 1):
        rows.append(model.fc1.weight.detach().clone())
        if step < steps:
            optimizer.zero_grad()
            loss = loss_fn(model(X), Y_BINARY)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        final_probabilities = torch.sigmoid(model(X)).squeeze(1)

    return rows, loss_fn(model(X), Y_BINARY).item(), final_probabilities


def activation_experiment():
    # Same seed and training settings for a controlled comparison.
    results = {}
    for name in ["sigmoid", "tanh", "relu"]:
        results[name] = train_binary(
            hidden_activation=name,
            seed=42,
            steps=5000,
            lr=1.0,
        )
    return results


def train_three_class(hidden_activation="tanh", seed=42, steps=3000, lr=1.0):
    torch.manual_seed(seed)
    model = ThreeClassNet(hidden_activation)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_loss = loss_fn(model(X), Y_THREE).item()

    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y_THREE)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    return {
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": loss.item(),
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
        "correct": int((predictions == Y_THREE).sum()),
    }


def main():
    print("=" * 70)
    print("TASK 1: XOR DATASET")
    print("=" * 70)
    print("X =\n", X)
    print("Y =\n", Y_BINARY.squeeze(1))
    print("XOR is not linearly separable, so one affine layer + sigmoid cannot")
    print("represent the required XOR decision boundary exactly.")

    print("\n" + "=" * 70)
    print("TASK 4A/B: BINARY XOR LEARNING + BACKPROPAGATION")
    print("=" * 70)
    binary = train_binary("tanh", seed=42, steps=5000, lr=1.0)
    print(f"Initial loss: {binary['initial_loss']:.9f}")
    print(f"Final loss:   {binary['final_loss']:.9f}")
    print("Probabilities:", binary["probabilities"].numpy())
    print("Predictions:  ", binary["predictions"].numpy())
    print(f"Correct:      {binary['correct']}/4")
    print("First-layer gradient dL/dW1:")
    print(binary["gradient"])
    print(f"Early gradient L2 norm: {binary['gradient_norm']:.9f}")

    print("\n" + "=" * 70)
    print("TASK 4C: ZERO-INITIALIZATION SYMMETRY")
    print("=" * 70)
    rows, sym_loss, sym_probs = symmetry_experiment("tanh", steps=5, lr=1.0)
    for step, weights in enumerate(rows):
        print(f"Step {step}:")
        print(weights)
    print("The two hidden rows remain identical.")
    print(f"Final loss: {sym_loss:.9f}")
    print("Final probabilities:", sym_probs.numpy())

    print("\n" + "=" * 70)
    print("TASK 4D: ACTIVATION EXPERIMENT")
    print("=" * 70)
    results = activation_experiment()
    print(f"{'Activation':<10} {'Final loss':>14} {'4/4 correct':>14} {'Early ||grad||2':>18}")
    for name, r in results.items():
        print(
            f"{name:<10} {r['final_loss']:>14.9f} "
            f"{r['correct'] == 4!s:>14} {r['gradient_norm']:>18.9f}"
        )

    print("\n" + "=" * 70)
    print("TASK 5: THREE-CLASS EXTENSION")
    print("=" * 70)
    three = train_three_class("tanh", seed=42, steps=3000, lr=1.0)
    print(f"Initial loss: {three['initial_loss']:.9f}")
    print(f"Final loss:   {three['final_loss']:.9f}")
    print("Logits:")
    print(three["logits"])
    print("Softmax probabilities:")
    print(three["probabilities"])
    print("Predictions:", three["predictions"].numpy())
    print(f"Correct: {three['correct']}/4")
    example = three["probabilities"][1]
    print("Example probability vector:", example.numpy())
    print("Sum of example probabilities:", example.sum().item())
    shifted = torch.softmax(three["logits"][1] + 100.0, dim=0)
    print("After adding 100 to all logits:", shifted.numpy())
    print("Output weight shape:", tuple(three["model"].fc2.weight.shape))


if __name__ == "__main__":
    main()
