import numpy as np
import matplotlib.pyplot as plt

# === 1. Generate Training Data (Hat shape) ===
x = np.linspace(-np.pi, np.pi, 400)
trainX = np.vstack((x, np.sinc(x)))  # shape (2, 400)

# === 2. SOM Parameters ===
num_neurons = 40
epochs = 100
learning_rate = 0.1
sigma_start = num_neurons / 2  # Initial neighborhood width
sigma_end = 1  # Final neighborhood width

# === 3. Initialize SOM Weights (2D weights for each neuron) ===
weights = np.random.uniform(low=-1, high=1, size=(2, num_neurons))

# === 4. Training Loop ===
for epoch in range(epochs):
    # Decay learning rate and neighborhood
    lr = learning_rate * (1 - epoch / epochs)
    sigma = sigma_start * ((sigma_end / sigma_start) ** (epoch / epochs))

    for i in range(trainX.shape[1]):
        x_i = trainX[:, i].reshape(2, 1)  # single 2D training point

        # Compute distances to each neuron
        dists = np.linalg.norm(weights - x_i, axis=0)

        # Best Matching Unit (BMU)
        bmu_idx = np.argmin(dists)

        # Neighborhood function
        neuron_indices = np.arange(num_neurons)
        dist_to_bmu = np.abs(neuron_indices - bmu_idx)
        h = np.exp(-(dist_to_bmu**2) / (2 * sigma**2))  # 1D Gaussian neighborhood

        # Update weights
        weights += lr * h * (x_i - weights)

# === 5. Visualization ===
plt.figure(figsize=(10, 6))
# Plot training data (hat curve)
plt.plot(trainX[0, :], trainX[1, :], '+r', label='Training Data')

# Plot neuron weights
plt.plot(weights[0, :], weights[1, :], 'bo-', label='SOM Neurons')

# Connect each neuron to its topological neighbors
for i in range(num_neurons - 1):
    x_coords = [weights[0, i], weights[0, i + 1]]
    y_coords = [weights[1, i], weights[1, i + 1]]
    plt.plot(x_coords, y_coords, 'b-')  # connection line

plt.title('Self-Organizing Map for Hat Function (sinc)')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
