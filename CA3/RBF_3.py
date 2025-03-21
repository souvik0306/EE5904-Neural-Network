import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 1. Prepare training and test sets
x_train = np.arange(-1.6, 1.6 + 0.08, 0.08)
n_train = len(x_train)
noise = 0.3 * np.random.randn(n_train)
y_train = 1.2 * np.sin(np.pi * x_train) - np.cos(2.4 * np.pi * x_train) + noise

x_test = np.arange(-1.6, 1.6 + 0.01, 0.01)
y_test = 1.2 * np.sin(np.pi * x_test) - np.cos(2.4 * np.pi * x_test)

# 2. Gaussian RBF
def gaussian_rbf(x, centers, sigma):
    return np.exp(-cdist(x[:, None], centers[:, None], 'sqeuclidean') / (2 * sigma**2))

centers = x_train.copy()
sigma = 0.1
Phi_train = gaussian_rbf(x_train, centers, sigma)
Phi_test = gaussian_rbf(x_test, centers, sigma)

# 3. Regularization: test multiple λ values
lambdas = [0, 0.001, 0.01, 0.1, 1, 10]
mse_list = []

# plt.figure(figsize=(12, 6))

for i, lam in enumerate(lambdas):
    # Ridge Regression solution
    w = np.linalg.solve(Phi_train.T @ Phi_train + lam * np.eye(len(centers)), Phi_train.T @ y_train)
    y_pred = Phi_test @ w
    mse = np.mean((y_test - y_pred)**2)
    mse_list.append(mse)

    # Plot each prediction
    # plt.plot(x_test, y_pred, label=f'λ={lam}, MSE={mse:.4f}')

# 4. Plot results
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.ravel()

for i, lam in enumerate(lambdas):
    w = np.linalg.solve(Phi_train.T @ Phi_train + lam * np.eye(len(centers)), Phi_train.T @ y_train)
    y_pred = Phi_test @ w
    mse = np.mean((y_test - y_pred)**2)
    
    axs[i].plot(x_test, y_test, 'k--', label='True Function')
    axs[i].plot(x_test, y_pred, 'r-', label=f'RBFN (λ={lam})')
    axs[i].set_title(f"λ = {lam} | MSE = {mse:.4f}")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig("image3_RBFN_Regularization.png", dpi=600)
plt.show()


# # 5. Plot MSE vs λ (log scale)
# plt.figure()
# plt.semilogx(lambdas, mse_list, marker='o')
# plt.xlabel("Regularization Factor (λ)")
# plt.ylabel("Test MSE")
# plt.title("Effect of Regularization on RBFN Performance")
# plt.grid(True)
# plt.savefig("image4_RBFN_Regularization_MSE.png", dpi=600)
# plt.show()

# for lam, mse in zip(lambdas, mse_list):
#     print(f"λ={lam:<5} → Test MSE = {mse:.5f}")
