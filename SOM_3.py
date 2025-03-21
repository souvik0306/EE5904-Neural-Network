import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import Counter

# === 1. Load data ===
data = scipy.io.loadmat('CA3/Digits.mat')
X_train_all = data['train_data'].T  # 1000 x 784
y_train_all = data['train_classlabel'].flatten()
X_test_all = data['test_data'].T
y_test_all = data['test_classlabel'].flatten()

# === 2. Omit classes based on matric number ===
omit_classes = [1, 2]  # from your matric number rule
mask_train = ~np.isin(y_train_all, omit_classes)
mask_test = ~np.isin(y_test_all, omit_classes)
X_train = X_train_all[mask_train]
y_train = y_train_all[mask_train]
X_test = X_test_all[mask_test]
y_test = y_test_all[mask_test]

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# === 3. SOM Parameters ===
grid_size = (10, 10)
epochs = 1000
eta0 = 0.1
sigma0 = max(grid_size) / 2
tau1 = epochs / np.log(sigma0)
tau2 = epochs

# === 4. Initialize SOM Weights ===
weights = np.random.rand(grid_size[0], grid_size[1], 784)  # each neuron has 784 weights

# === 5. SOM Training ===
for epoch in range(epochs):
    eta = eta0 * np.exp(-epoch / tau2)
    sigma = sigma0 * np.exp(-epoch / tau1)

    # Optional progress bar or batch shuffle here

    for idx, x in enumerate(X_train):
        dists = np.linalg.norm(weights - x, axis=2)
        bmu_idx = np.unravel_index(np.argmin(dists), grid_size)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                grid_dist = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                h = np.exp(-(grid_dist ** 2) / (2 * sigma ** 2))
                weights[i, j] += eta * h * (x - weights[i, j])

    # --- Print progress every 100 epochs ---
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs} — η = {eta:.5f}, σ = {sigma:.5f}")

# === 6. Semantic Map (Winning Neuron Class Counts) ===
semantic_map = [[[] for _ in range(grid_size[1])] for _ in range(grid_size[0])]

for x, label in zip(X_train, y_train):
    dists = np.linalg.norm(weights - x, axis=2)
    i, j = np.unravel_index(np.argmin(dists), grid_size)
    semantic_map[i][j].append(label)

# Get majority class at each neuron
semantic_class_map = np.zeros(grid_size, dtype=int) - 1  # -1 = no winner

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        if semantic_map[i][j]:
            counts = Counter(semantic_map[i][j])
            semantic_class_map[i, j] = counts.most_common(1)[0][0]

# === 7. Plot Semantic Map ===
plt.figure(figsize=(8, 8))
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        label = semantic_class_map[i, j]
        if label != -1:
            plt.text(j, i, str(label), ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue'))
        else:
            plt.text(j, i, '.', ha='center', va='center', fontsize=10)
plt.title("Semantic Map (Most Frequent Class per Neuron)")
plt.xlim(-0.5, 9.5)
plt.ylim(-0.5, 9.5)
plt.gca().invert_yaxis()
plt.grid(True)
plt.savefig("image10_SOM_SemanticMap.png", dpi=600)
plt.show()

# === 8. Visualize SOM Neuron Weights ===
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        ax = axes[i, j]
        ax.imshow(weights[i, j].reshape(28, 28), cmap='gray')
        ax.axis('off')
plt.suptitle("SOM Neuron Weight Visualization", y=1.01)
plt.savefig("image11_SOM_Weights.png", dpi=600)
plt.tight_layout()
plt.show()
