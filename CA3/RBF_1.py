import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 1. Generate training data
x_train = np.arange(-1.6, 1.6 + 0.08, 0.08)
n_train = len(x_train)
noise = 0.3 * np.random.randn(n_train)
y_train = 1.2 * np.sin(np.pi * x_train) - np.cos(2.4 * np.pi * x_train) + noise

# 2. Generate test data
x_test = np.arange(-1.6, 1.6 + 0.01, 0.01)
y_test = 1.2 * np.sin(np.pi * x_test) - np.cos(2.4 * np.pi * x_test)

# 3. Gaussian RBF function
def gaussian_rbf(x, c, sigma=0.1):
    return np.exp(-cdist(x[:, None], c[:, None], 'sqeuclidean') / (2 * sigma**2))

# 4. Build Φ matrix (interpolation matrix)
Phi = gaussian_rbf(x_train, x_train, sigma=0.1)

# 5. Solve for weights (exact interpolation)
weights = np.linalg.solve(Phi, y_train)

# 6. Predict on test set
Phi_test = gaussian_rbf(x_test, x_train, sigma=0.1)
y_pred = Phi_test @ weights

# 7. Plot and evaluate
plt.figure(figsize=(10, 5))
plt.plot(x_test, y_test, label='True Function', color='blue')
plt.plot(x_test, y_pred, label='RBFN Prediction', color='red', linestyle='--')
plt.scatter(x_train, y_train, label='Noisy Training Points', color='black', s=10)
plt.legend()
plt.title("RBFN Exact Interpolation - Function Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig("image1_RBFN_Exact_Interpolation.png", dpi=600)
plt.show()

# 8. Compute MSE
mse = np.mean((y_test - y_pred) ** 2)
print(f"Test Set Mean Squared Error: {mse:.4f}")
