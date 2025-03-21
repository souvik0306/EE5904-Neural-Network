import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# === 1. Load the data ===
data = scipy.io.loadmat('CA3/mnist_m.mat')

X_train = data['train_data'] / 255.0    # Normalize like MATLAB
y_train = data['train_classlabel'].flatten()

X_test = data['test_data'] / 255.0
y_test = data['test_classlabel'].flatten()

# === 2. Create binary labels: class 4 and 6 → 1, rest → 0 ===
TrLabel = np.isin(y_train, [4, 6]).astype(int)
TeLabel = np.isin(y_test, [4, 6]).astype(int)

TrN = len(TrLabel)
TeN = len(TeLabel)

# === 3. Gaussian RBF Function ===
def gaussian_rbf(X, centers, sigma):
    D = cdist(X.T, centers.T, 'sqeuclidean')
    return np.exp(-D / (2 * sigma**2))

# === 4. Exact Interpolation with Regularization ===
def train_rbf_exact(X_train, y_train, sigma, lam):
    Phi = gaussian_rbf(X_train, X_train, sigma)
    I = np.eye(Phi.shape[1])
    W = np.linalg.solve(Phi.T @ Phi + lam * I, Phi.T @ y_train)
    return W, X_train

# === 5. Prediction ===
def predict_rbf(X, centers, weights, sigma):
    Phi = gaussian_rbf(X, centers, sigma)
    return Phi @ weights

# === 6. Threshold Accuracy Calculation ===
def threshold_sweep(pred, label, num_steps=1000):
    TrAcc = np.zeros(num_steps)
    thresholds = np.linspace(pred.min(), pred.max(), num_steps)
    for i, t in enumerate(thresholds):
        pred_bin = (pred >= t).astype(int)
        correct = np.sum((pred_bin == label))
        TrAcc[i] = correct / len(label)
    return thresholds, TrAcc

# === 7. Loop for Different λ Values ===
lambda_values = [0.001, 0.01, 0.1, 1, 10, 100]
sigma = 100

for lam in lambda_values:
    # Train
    weights, centers = train_rbf_exact(X_train, TrLabel, sigma, lam)
    
    # Predict
    TrPred = predict_rbf(X_train, centers, weights, sigma)
    TePred = predict_rbf(X_test, centers, weights, sigma)

    # Threshold sweep
    thresholds, TrAcc = threshold_sweep(TrPred, TrLabel)
    _, TeAcc = threshold_sweep(TePred, TeLabel)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, TrAcc, label='Train Accuracy', linestyle='-', color='blue', linewidth=2)
    plt.plot(thresholds, TeAcc, label='Test Accuracy', linestyle='--', color='red', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title(f'Regularization (lambda = {lam})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
