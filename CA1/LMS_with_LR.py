import numpy as np
import matplotlib.pyplot as plt

X = np.array([0, 0.8, 1.6, 3, 4.0, 5.0])
y = np.array([0.5, 1, 4, 5, 6, 8])

# Part (a): Solve for w and b using the Linear Least Squares (LLS) method
X_matrix = np.vstack([X, np.ones(len(X))]).T  # Add bias term
w_lls, b_lls = np.linalg.lstsq(X_matrix, y, rcond=None)[0]  # Solve using LLS

# Generate predictions
X_fit = np.linspace(min(X), max(X), 100)
y_fit_lls = w_lls * X_fit + b_lls

# Part (b): Solve for w and b using the Least Mean Squares (LMS) Algorithm
def lms_learning(X, y, learning_rate=0.01, epochs=100):
    """
    Train a single-layer perceptron using the LMS (Least Mean Squares) rule.
    Returns the weight and bias updates over epochs.
    """
    w = np.random.randn()  # Initialize weight randomly
    b = np.random.randn()  # Initialize bias randomly
    weight_trajectory = [(w, b)]  # Store weight updates

    for epoch in range(epochs):
        for i in range(len(X)):
            y_pred = w * X[i] + b  # Compute prediction
            error = y[i] - y_pred  # Compute error

            # Weight update rule
            w += learning_rate * error * X[i]
            b += learning_rate * error

        weight_trajectory.append((w, b))

    return weight_trajectory

# Train using LMS algorithm
learning_rate = 0.01
epochs = 100
weight_updates_lms = lms_learning(X, y, learning_rate, epochs)

# Extract weight trajectory
epochs_list = np.arange(len(weight_updates_lms))
w_vals = [w[0] for w in weight_updates_lms]
b_vals = [w[1] for w in weight_updates_lms]


# Generate LMS fitted line using final w, b
w_lms_final, b_lms_final = weight_updates_lms[-1]
y_fit_lms = w_lms_final * X_fit + b_lms_final

# # Plot LMS fitting
# plt.figure(figsize=(6, 5))
# plt.scatter(X, y, color='red', label="Data Points")
# plt.plot(X_fit, y_fit_lls, label="LLS Fit", linestyle="dashed", color="blue")
# plt.plot(X_fit, y_fit_lms, label=f"LMS Fit: y = {w_lms_final:.4f}x + {b_lms_final:.4f}", color="green", linewidth=2)

# # Highlight the differences between LLS and LMS fits
# plt.fill_between(X_fit, y_fit_lls, y_fit_lms, color='gray', alpha=0.2, label="Difference Area")

# plt.xlabel("X")
# plt.ylabel("y")
# plt.title("Linear Fit using LMS Algorithm (Learning Rate = 0.01)")
# plt.legend()
# plt.grid(True)
# plt.savefig("images4_LMS_LLS.png", dpi=600)
# plt.show()
# # Plot weight updates over epochs
# plt.figure(figsize=(9, 5))
# plt.plot(epochs_list, w_vals, label="w", marker='o', color="blue")
# plt.plot(epochs_list, b_vals, label="b", marker='s', color="green")

# # Annotate final values
# plt.annotate(f'w = {w_vals[-1]:.4f}', xy=(epochs_list[-1], w_vals[-1]), xytext=(epochs_list[-1], w_vals[-1] + 0.5),
#              arrowprops=dict(facecolor='blue', shrink=0.05))
# plt.annotate(f'b = {b_vals[-1]:.4f}', xy=(epochs_list[-1], b_vals[-1]), xytext=(epochs_list[-1], b_vals[-1] + 0.5),
#              arrowprops=dict(facecolor='green', shrink=0.05))

# plt.xlabel("Epochs")
# plt.ylabel("Values")
# plt.title("Weight and Bias Updates over 100 epochs using LMS Algorithm")
# plt.legend()
# plt.grid(True)
# plt.savefig("images4_LMS.png", dpi=600)
# plt.show()

# # Print final LMS results
# print(f"Solution using LMS method after {epochs} epochs: w = {w_lms_final:.4f}, b = {b_lms_final:.4f}")

# Part (d): Repeat LMS with different learning rates and analyze convergence
learning_rates = [0.001, 0.01, 0.1, 1.0]

plt.figure(figsize=(12, 6))
for i, lr in enumerate(learning_rates):
    weight_updates = lms_learning(X, y, learning_rate=lr, epochs=100)

    # Extract weight trajectory
    epochs_list = np.arange(len(weight_updates))
    w_vals = [w[0] for w in weight_updates]
    b_vals = [w[1] for w in weight_updates]

    # Plot weight updates
    plt.subplot(2, 2, i + 1)
    plt.plot(epochs_list, w_vals, label="w", marker='o', color="blue")
    plt.plot(epochs_list, b_vals, label="b", marker='s', color="green")
    
    # Annotate final values
    plt.annotate(f'w = {w_vals[-1]:.4f}', xy=(epochs_list[-1], w_vals[-1]), xytext=(epochs_list[-1], w_vals[-1] + 0.5),
                 arrowprops=dict(facecolor='blue', shrink=0.05))
    plt.annotate(f'b = {b_vals[-1]:.4f}', xy=(epochs_list[-1], b_vals[-1]), xytext=(epochs_list[-1], b_vals[-1] + 0.5),
                 arrowprops=dict(facecolor='green', shrink=0.05))
    
    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.title(f"Learning Rate: {lr}")
    plt.grid(True)

# Add common legend
plt.figlegend(["w", "b"], loc='upper left')
plt.tight_layout()
plt.savefig("images4_LMS_LearningRates.png", dpi=600)
plt.show()