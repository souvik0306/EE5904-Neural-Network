import numpy as np
import matplotlib.pyplot as plt

# Perceptron learning algorithm function
def perceptron_learning(X, y, learning_rate, max_epochs=20, seed=59):
    """
    Train a perceptron for a given logic function using a specified learning rate.
    Returns the weight trajectory.
    """
    np.random.seed(seed)  # Set seed for consistent results
    num_features = X.shape[1]
    w = np.random.randn(num_features)  # Random weight initialization
    b = np.random.randn()  # Random bias initialization
    weight_trajectory = [(w.copy(), b)]  # Store weight updates

    for _ in range(max_epochs):
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

# Seed value for reproducibility
seed_value = 34

# Store weight updates for each gate and learning rate
stored_weight_updates = {}

# Train perceptrons and store weight updates
for gate, (X, y) in logic_gates.items():
    stored_weight_updates[gate] = {}
    for lr in learning_rates:
        stored_weight_updates[gate][lr] = perceptron_learning(X, y, lr, seed=seed_value)

## Function to plot weight updates for a given gate
def plot_weight_updates(gate, learning_rates):
    fig, axes = plt.subplots(1, len(learning_rates), figsize=(15, 4))

    for col, lr in enumerate(learning_rates):
        weight_updates = stored_weight_updates[gate][lr]  # Use stored values

        # Extract weight trajectory
        epochs = np.arange(len(weight_updates))
        w1_vals = [w[0][0] if isinstance(w[0], np.ndarray) and len(w[0]) > 1 else w[0] for w in weight_updates]
        w2_vals = [w[0][1] if isinstance(w[0], np.ndarray) and len(w[0]) > 1 else 0 for w in weight_updates]
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
    plt.savefig(f"{gate}_weight_updates.png", dpi=600)  # Save the figure
    plt.show()

# Plot weight updates using stored values
for gate in logic_gates.keys():
    plot_weight_updates(gate, learning_rates)

# Function to print final weight values
def print_final_weights():
    print("\nFinal Learned Weights & Bias:")
    print("Gate\t\tLearning Rate\tw1\t\tw2\t\tBias")
    print("-" * 60)
    
    for gate, lr_dict in stored_weight_updates.items():
        for lr, weight_updates in lr_dict.items():
            final_w = weight_updates[-1][0]
            final_b = weight_updates[-1][1]
            if isinstance(final_w, np.ndarray) and len(final_w) > 1:
                w1 = final_w[0].item()  # Convert to scalar
                w2 = final_w[1].item()  # Convert to scalar
            else:
                w1 = final_w.item()  # Convert to scalar
                w2 = 0
            print(f"{gate}\t\t{lr:.2f}\t\t{w1:.4f}\t{w2:.4f}\t{final_b:.4f}")

# Print the final weight and bias values
print_final_weights()

# Function to plot final weights
def plot_final_weights():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, param in enumerate(["w1", "w2", "Bias"]):
        for gate, lr_dict in stored_weight_updates.items():
            x_vals = []
            y_vals = []
            for lr, weight_updates in lr_dict.items():
                final_w = weight_updates[-1][0]
                final_b = weight_updates[-1][1]
                if param == "w1":
                    val = final_w[0] if isinstance(final_w, np.ndarray) and len(final_w) > 1 else final_w
                elif param == "w2":
                    val = final_w[1] if isinstance(final_w, np.ndarray) and len(final_w) > 1 else 0
                else:  # Bias
                    val = final_b
                
                x_vals.append(lr)
                y_vals.append(val)

            axes[idx].plot(x_vals, y_vals, marker='o', label=gate)

        axes[idx].set_title(f"Final {param} Values")
        axes[idx].set_xlabel("Learning Rate")
        axes[idx].set_ylabel(param)
        axes[idx].set_xscale("log")
        axes[idx].grid(True)
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig("final_weights.png")  # Save the figure
    plt.show()

# Plot final learned weights and bias
plot_final_weights()