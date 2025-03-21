import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 1. Generate training and test data
x_train = np.arange(-1.6, 1.6 + 0.08, 0.08)
n_train = len(x_train)
noise = 0.3 * np.random.randn(n_train)
y_train = 1.2 * np.sin(np.pi * x_train) - np.cos(2.4 * np.pi * x_train) + noise

x_test = np.arange(-1.6, 1.6 + 0.01, 0.01)
y_test = 1.2 * np.sin(np.pi * x_test) - np.cos(2.4 * np.pi * x_test)

# 2. Select 20 random centers from training points
np.random.seed(42)  # for reproducibility
num_centers = 20
center_indices = np.random.choice(n_train, num_centers, replace=False)
centers = x_train[center_indices]

# 3. RBF matrix function
def gaussian_rbf(x, centers, sigma):
    return np.exp(-cdist(x[:, None], centers[:, None], 'sqeuclidean') / (2 * sigma**2))

# 4. Build Φ matrix and solve weights using least squares
sigma = 0.1
Phi_train = gaussian_rbf(x_train, centers, sigma)
weights, _, _, _ = np.linalg.lstsq(Phi_train, y_train, rcond=None)

# 5. Predict on test set
Phi_test = gaussian_rbf(x_test, centers, sigma)
y_pred = Phi_test @ weights

# 6. Plot results
plt.figure(figsize=(12, 6))
plt.plot(x_test, y_test, label='True Function', color='blue', linewidth=3)
plt.plot(x_test, y_pred, '--', label='RBFN Approximation', color='red', linewidth=3)
plt.scatter(x_train, y_train, label='Noisy Training Data', s=10, color='black')
plt.scatter(centers, 1.2*np.sin(np.pi*centers) - np.cos(2.4*np.pi*centers), 
            color='green', s=80, marker='x', label='Selected Centers', linewidths=2)
plt.title("RBFN with 20 Random Centers")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig("image2_RBFN_20_Random_Centers.png", dpi=600)
plt.show()

# 7. Evaluate
mse_b = np.mean((y_test - y_pred)**2)
print(f"Test Set MSE (20 random centers): {mse_b:.4f}")
