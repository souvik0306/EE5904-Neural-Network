import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# --- Load Data ---
data = loadmat('CA3/mnist_m.mat')
X_train_all = data['train_data'] / 255.0
y_train_all = data['train_classlabel'].flatten()
X_test_all = data['test_data'] / 255.0
y_test_all = data['test_classlabel'].flatten()

# --- Select classes 4 and 6 ---
train_idx = np.where((y_train_all == 4) | (y_train_all == 6))[0]
test_idx = np.where((y_test_all == 4) | (y_test_all == 6))[0]
X_train = X_train_all[:, train_idx]
X_test = X_test_all[:, test_idx]
TrLabel = (y_train_all[train_idx] == 4).astype(int)
TeLabel = (y_test_all[test_idx] == 4).astype(int)

# --- Step 1: Apply KMeans with 2 clusters ---
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train.T)
centers = kmeans.cluster_centers_.T  # shape: (784, 2)

# --- Step 2: Compute distances and RBF matrix ---
def gaussian_rbf(r2, sigma):
    return np.exp(-r2 / (2 * sigma**2))

# Estimate σ from center spread (max distance between centers and samples)
d2 = cdist(X_train.T, centers.T, 'sqeuclidean')
sigma = np.sqrt(np.max(d2)) / np.sqrt(2 * 2)

# Construct Φ matrices
Phi_train = gaussian_rbf(d2, sigma)
Phi_test = gaussian_rbf(cdist(X_test.T, centers.T, 'sqeuclidean'), sigma)

# --- Step 3: Solve for weights ---
w = np.linalg.pinv(Phi_train) @ TrLabel

# --- Step 4: Predict and Evaluate ---
TrPred = Phi_train @ w
TePred = Phi_test @ w
TrAcc = np.mean((TrPred >= 0.5) == TrLabel)
TeAcc = np.mean((TePred >= 0.5) == TeLabel)

print(f"Train Accuracy with 2 K-Means Centers: {TrAcc:.4f}")
print(f"Test Accuracy with 2 K-Means Centers: {TeAcc:.4f}")

# --- Step 5: Compute mean images for class 4 and 6 ---
class4_mean = np.mean(X_train[:, TrLabel == 1], axis=1)
class6_mean = np.mean(X_train[:, TrLabel == 0], axis=1)

# --- Step 6: Visualize centers and class means ---
def show_image(img, title, subplot_idx):
    plt.subplot(2, 2, subplot_idx)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(8, 9))  # Slightly taller to fit title
show_image(centers[:, 0], "KMeans Center 1", 1)
show_image(centers[:, 1], "KMeans Center 2", 2)
show_image(class4_mean, "Mean Image: Class 4", 3)
show_image(class6_mean, "Mean Image: Class 6", 4)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for suptitle
plt.suptitle("KMeans Centers vs Class Means", fontsize=14)
plt.savefig("images7_KMeans_vs_ClassMeans.png", dpi=600)
plt.show()

# === Threshold Sweep for Performance Visualization ===
def threshold_sweep(pred, label, num_points=1000):
    thresholds = np.linspace(pred.min(), pred.max(), num_points)
    acc = np.zeros(num_points)
    for i, t in enumerate(thresholds):
        pred_bin = (pred >= t).astype(int)
        acc[i] = np.mean(pred_bin == label)
    return thresholds, acc

thr, TrAcc = threshold_sweep(TrPred, TrLabel)
_, TeAcc = threshold_sweep(TePred, TeLabel)

# === Plot the Performance ===
plt.figure(figsize=(8, 4))
plt.plot(thr, TrAcc, '--', label='Train Accuracy', color='blue', linewidth=2)
plt.plot(thr, TeAcc, '--', label='Test Accuracy', color='red', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title("Accuracy vs Threshold (2 KMeans Centers)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("image8_KMeans_ThresholdSweep.png", dpi=600)
plt.show()
