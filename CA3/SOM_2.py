import numpy as np
import matplotlib.pyplot as plt

# === 1. Generate Circle Data ===
np.random.seed(0)
X = np.random.randn(800, 2)
s2 = np.sum(X**2, axis=1)
scale = (1 * (np.power(np.random.gamma(shape=1, scale=1, size=800), 0))) / np.sqrt(s2)
trainX = (X.T * scale).T.T  # shape (2, 800)

# === 2. SOM Parameters ===
grid_size = (8, 8)
num_neurons = grid_size[0] * grid_size[1]
epochs = 500
learning_rate = 0.3
sigma_start = max(grid_size) / 2
sigma_end = 1

# === 3. Initialize SOM Weights ===
weights = np.random.uniform(low=-1, high=1, size=(2, grid_size[0], grid_size[1]))

# === 4. Get neuron coordinate grid ===
neuron_coords = np.array([(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])]).reshape(grid_size[0], grid_size[1], 2)

# === 5. Training ===
epochs = 500
eta0 = 0.1
sigma0 = max(grid_size) / 2
tau1 = epochs / np.log(sigma0)
tau2 = epochs

for epoch in range(epochs):
    lr = eta0 * np.exp(-epoch / tau2)
    sigma = sigma0 * np.exp(-epoch / tau1)
    
    for i in range(trainX.shape[1]):
        x_i = trainX[:, i].reshape(2, 1)

        # Compute distances to all neurons
        dists = np.linalg.norm(weights.reshape(2, -1) - x_i, axis=0)
        bmu_index = np.argmin(dists)
        bmu_i, bmu_j = np.unravel_index(bmu_index, grid_size)

        for r in range(grid_size[0]):
            for c in range(grid_size[1]):
                d_grid = np.linalg.norm(np.array([r, c]) - np.array([bmu_i, bmu_j]))
                h = np.exp(-(d_grid**2) / (2 * sigma**2))
                weights[:, r, c] += lr * h * (x_i.flatten() - weights[:, r, c])

# === 6. Plotting the SOM Map ===
plt.figure(figsize=(8, 8))

# Plot the training data (red '+')
plt.plot(trainX[0, :], trainX[1, :], '+r', label='Training Data')

# Plot neuron weights
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        x, y = weights[0, i, j], weights[1, i, j]
        plt.plot(x, y, 'bo')

        # Connect to right neighbor
        if j + 1 < grid_size[1]:
            x2, y2 = weights[0, i, j+1], weights[1, i, j+1]
            plt.plot([x, x2], [y, y2], 'b-')

        # Connect to bottom neighbor
        if i + 1 < grid_size[0]:
            x2, y2 = weights[0, i+1, j], weights[1, i+1, j]
            plt.plot([x, x2], [y, y2], 'b-')

plt.title("2D SOM (8x8) Mapping to Circle")
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.savefig("image9_SOM_Circle.png", dpi=600)
plt.show()
