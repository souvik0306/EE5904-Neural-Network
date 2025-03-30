import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial.distance import cdist

# --- Load data ---
data = scipy.io.loadmat('mnist_m.mat')
X_train_all = data['train_data'] / 255.0
y_train_all = data['train_classlabel'].flatten()
X_test_all = data['test_data'] / 255.0
y_test_all = data['test_classlabel'].flatten()

# Select classes 4 and 6
train_idx = np.where((y_train_all == 4) | (y_train_all == 6))[0]
test_idx = np.where((y_test_all == 4) | (y_test_all == 6))[0]
X_train = X_train_all[:, train_idx]
X_test = X_test_all[:, test_idx]
TrLabel = (y_train_all[train_idx] == 4).astype(int)
TeLabel = (y_test_all[test_idx] == 4).astype(int)

# --- Distance & RBF Functions ---
def compute_squared_distances(X1, X2):
    return cdist(X1.T, X2.T, 'sqeuclidean')

def gaussian_rbf(r2, sigma):
    return np.exp(-r2 / (2 * sigma**2))

def threshold_sweep(pred, label, num_points=1000):
    thresholds = np.linspace(pred.min(), pred.max(), num_points)
    acc = np.zeros(num_points)
    for i, t in enumerate(thresholds):
        pred_bin = (pred >= t).astype(int)
        acc[i] = np.mean(pred_bin == label)
    return thresholds, acc

# --- Randomly select 100 centers ---
np.random.seed(520)
num_centers = 100
rand_indices = np.random.choice(X_train.shape[1], num_centers, replace=False)
centers = X_train[:, rand_indices]

# --- Sigma values to test ---
sigma_values = [-1, 0.1, 1, 10, 100, 1000]

# --- Create subplot grid ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for idx, sigma in enumerate(sigma_values):
    # Distance computation
    D_train = compute_squared_distances(X_train, centers)
    D_test = compute_squared_distances(X_test, centers)

    # Auto width
    sigma_str = f'{sigma:.2f}'
    sigma_val = sigma

    # RBF computation
    Phi_train = gaussian_rbf(D_train, sigma_val)
    Phi_test = gaussian_rbf(D_test, sigma_val)

    # Solve weights
    w = np.linalg.pinv(Phi_train) @ TrLabel

    # Predict
    TrPred = Phi_train @ w
    TePred = Phi_test @ w

    # Sweep threshold
    thr, TrAcc = threshold_sweep(TrPred, TrLabel)
    _, TeAcc = threshold_sweep(TePred, TeLabel)

    # Plot
    ax = axes[idx]
    ax.plot(thr, TrAcc, label='Train Acc', linestyle='-', markersize=2, color='blue')
    ax.plot(thr, TeAcc, label='Test Acc', linestyle='--', markersize=2, color='red')
    ax.set_title(f'σ = {sigma_str}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.legend()

# Hide any extra subplots if not used
for j in range(len(sigma_values), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("image6_MNIST_RBFN.png", dpi=600)
plt.show()

# === Q2(a): Exact Interpolation ===
Phi_a = gaussian_rbf(compute_squared_distances(X_train, X_train), 100)
w_a = np.linalg.pinv(Phi_a) @ TrLabel
TrPred_a = Phi_a @ w_a
TePred_a = gaussian_rbf(compute_squared_distances(X_test, X_train), 100) @ w_a
a_train_acc = np.mean((TrPred_a >= 0.5) == TrLabel)
a_test_acc = np.mean((TePred_a >= 0.5) == TeLabel)

# === Q2(b): Fixed 100 Centers ===
Phi_b = gaussian_rbf(compute_squared_distances(X_train, centers), 100)
w_b = np.linalg.pinv(Phi_b) @ TrLabel
TrPred_b = Phi_b @ w_b
TePred_b = gaussian_rbf(compute_squared_distances(X_test, centers), 100) @ w_b
b_train_acc = np.mean((TrPred_b >= 0.5) == TrLabel)
b_test_acc = np.mean((TePred_b >= 0.5) == TeLabel)

print("=== Comparison at σ = 100, Threshold = 0.5 ===")
print(f"Q2(a) Exact Interpolation → Train Acc: {a_train_acc:.4f}, Test Acc: {a_test_acc:.4f}")
print(f"Q2(b) 100 Random Centers  → Train Acc: {b_train_acc:.4f}, Test Acc: {b_test_acc:.4f}")
