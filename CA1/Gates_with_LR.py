import numpy as np
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(59)

# Perceptron learning algorithm function
def perceptron_learning(X, y, learning_rate, max_epochs=20):
    """
    Train a perceptron for a given logic function using specified learning rate.
    Returns the weight trajectory.
    """
    num_features = X.shape[1]
    w = np.random.randn(num_features)  # Random weight initialization
    b = np.random.randn()  # Random bias initialization
    weight_trajectory = [(w.copy(), b)]  # Store weight updates

    for epoch in range(max_epochs):
        v = np.dot(X, w) + b
        y_pred = np.where(v >= 0, 1, 0)
        errors = y - y_pred

        if np.any(errors != 0):
            w += learning_rate * np.dot(errors, X)
            b += learning_rate * np.sum(errors)
            weight_trajectory.append((w.copy(), b))

    return weight_trajectory

# Define logic gates and their truth tables
logic_gates = {
    "AND": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])),
    "OR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])),
    "NAND": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([1, 1, 1, 0])),
    "COMPLEMENT": (np.array([[0], [1]]), np.array([1, 0]))
}

# Define different learning rates to experiment with
learning_rates = [0.01, 0.1, 1.0, 5.0]

# Function to plot weight updates for a given gate
def plot_weight_updates(gate, X, y, learning_rates):
    fig, axes = plt.subplots(1, len(learning_rates), figsize=(15, 4))
    for col, lr in enumerate(learning_rates):
        weight_updates = perceptron_learning(X, y, lr)
        
        # Extract weight trajectory
        epochs = np.arange(len(weight_updates))
        w1_vals = [w[0][0] if len(w[0]) > 1 else w[0] for w in weight_updates]
        w2_vals = [w[0][1] if len(w[0]) > 1 else 0 for w in weight_updates]
        b_vals = [w[1] for w in weight_updates]

        # Plot weight updates
        axes[col].plot(epochs, w1_vals, label="w1", marker='o')
        axes[col].plot(epochs, w2_vals, label="w2", marker='s')
        axes[col].plot(epochs, b_vals, label="Bias", marker='^')

        # Formatting
        axes[col].set_title(f"Learning Rate: {lr}")
        axes[col].set_xlabel("Epochs")
        axes[col].set_ylabel("Weights & Bias")
        axes[col].grid(True)

    # Add common legend
    fig.legend(["w1", "w2", "Bias"], loc='upper right', ncol=3)
    fig.suptitle(f"Weight Updates for {gate} Gate")
    plt.tight_layout()
    plt.savefig(f"images4_{gate}.png", dpi=600)
    plt.show()

# Plot weight updates for each gate
for gate, (X, y) in logic_gates.items():
    plot_weight_updates(gate, X, y, learning_rates)


# Define the learning rate to experiment with
learning_rate = 1.0
# Function to print weight updates for a given gate
def print_weight_updates(gate, X, y, learning_rate):
    weight_updates = perceptron_learning(X, y, learning_rate)
    
    # Extract weight trajectory
    epochs = np.arange(len(weight_updates))
    w1_vals = [float(w[0][0]) if isinstance(w[0], np.ndarray) else float(w[0]) for w in weight_updates]
    w2_vals = [float(w[0][1]) if isinstance(w[0], np.ndarray) and len(w[0]) > 1 else 0 for w in weight_updates]
    b_vals = [float(w[1]) for w in weight_updates]

    print(f"Weight updates for {gate} gate with learning rate {learning_rate}:")
    print("Epoch\tw1\t\tw2\t\tBias")
    for epoch, (w1, w2, b) in enumerate(zip(w1_vals, w2_vals, b_vals)):
        print(f"{epoch}\t{w1:.4f}\t{w2:.4f}\t{b:.4f}")
    print()

# Print weight updates for each gate
for gate, (X, y) in logic_gates.items():
    print_weight_updates(gate, X, y, learning_rate)