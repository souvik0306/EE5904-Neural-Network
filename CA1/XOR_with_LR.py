# Re-import necessary libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
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

# Define the XOR truth table
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR outputs

# Train the perceptron on XOR using learning rate = 1.0
learning_rate = 1.0
weight_updates_xor = perceptron_learning(X_xor, y_xor, learning_rate)

# Extract weight trajectory
epochs_xor = np.arange(len(weight_updates_xor))
w1_vals_xor = [float(w[0][0]) for w in weight_updates_xor]
w2_vals_xor = [float(w[0][1]) for w in weight_updates_xor]
b_vals_xor = [float(w[1]) for w in weight_updates_xor]

# Plot weight updates for XOR
plt.figure(figsize=(8, 5))
plt.plot(epochs_xor, w1_vals_xor, label="w1", marker='o')
plt.plot(epochs_xor, w2_vals_xor, label="w2", marker='s')
plt.plot(epochs_xor, b_vals_xor, label="Bias", marker='^')

# Formatting
plt.title("Weight Updates for XOR Gate (Learning Rate: 1.0)")
plt.xlabel("Epochs")
plt.ylabel("Weights & Bias")
plt.legend()
plt.grid(True)
plt.savefig("images4_XOR.png", dpi=600)
plt.show()

# Print the weight updates
print("Weight updates for XOR gate with learning rate 1.0:")
print("Epoch\tw1\t\tw2\t\tBias")
for epoch, (w1, w2, b) in enumerate(zip(w1_vals_xor, w2_vals_xor, b_vals_xor)):
    print(f"{epoch}\t{w1:.4f}\t{w2:.4f}\t{b:.4f}")
