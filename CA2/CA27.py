import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load and Normalize Data
# -------------------------------
def load_images(folder_path, label):
    images, labels = [], []
    for i in range(500):  # 500 images per class
        img_path = os.path.join(folder_path, f"{i:03d}.jpg")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = img.flatten()  # Flatten to 1024
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Load both classes
animal_images, animal_labels = load_images(r"CA2/group_1/dog", label=1)
object_images, object_labels = load_images(r"CA2/group_1/automobile", label=0)

# Combine dataset
X = np.vstack((animal_images, object_images))
Y = np.hstack((animal_labels, object_labels))

# -------------------------------
# Step 2: Compute Global Mean and Variance
# -------------------------------
global_mean = np.mean(X, axis=0)
global_variance = np.var(X, axis=0)

# Normalize the dataset
X_normalized = (X - global_mean) / (np.sqrt(global_variance) + 1e-8)  # Prevent division by zero

# Ensure dimensions remain the same
print("Original Shape:", X.shape)  # Should be (1000, 1024)
print("Normalized Shape:", X_normalized.shape)  # Should be (1000, 1024)

# Shuffle the dataset randomly
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, Y = X[indices], Y[indices]

# Split datasets
split_index = int(0.9 * X.shape[0])  # 90% for training, 10% for validation
X_train, Y_train = X[:split_index], Y[:split_index]
X_val, Y_val = X[split_index:], Y[split_index:]

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=500):
        self.weights = np.random.randn(input_size + 1) * 0.01  # Add bias term
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.history = {"train_acc": [], "val_acc": []}

    def activation(self, x):
        return 1 if x >= 0 else 0

    def train(self, X_train, Y_train, X_val, Y_val):
        X_train = np.c_[X_train, np.ones(X_train.shape[0])]  # Add bias
        X_val = np.c_[X_val, np.ones(X_val.shape[0])]  # Add bias

        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                y_pred = self.activation(np.dot(self.weights, X_train[i]))
                self.weights += self.learning_rate * (Y_train[i] - y_pred) * X_train[i]

            train_acc = self.evaluate(X_train, Y_train)
            val_acc = self.evaluate(X_val, Y_val)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}/{self.epochs}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

    def evaluate(self, X, Y):
        predictions = np.array([self.activation(np.dot(self.weights, x)) for x in X])
        return np.mean(predictions == Y) * 100

# Train the perceptron on normalized data
perceptron = Perceptron(input_size=1024, learning_rate=0.001, epochs=1000)
perceptron.train(X_train, Y_train, X_val, Y_val)

# -------------------------------
# Step 3: Visualize Training and Validation Accuracy
# -------------------------------
# Smoothen the curves using a moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Define window size for smoothing
window_size = 50

# Apply moving average to smoothen the accuracy curves
smoothed_train_acc = moving_average(perceptron.history["train_acc"], window_size)
smoothed_val_acc = moving_average(perceptron.history["val_acc"], window_size)

# Adjust the range for the x-axis to match the smoothed data
epochs_range = range(perceptron.epochs - window_size + 1)

plt.figure(figsize=(7, 5))
plt.plot(epochs_range, smoothed_train_acc, label="Training Accuracy (Smoothed)", linewidth=4, color='red')
plt.plot(epochs_range, smoothed_val_acc, label="Validation Accuracy (Smoothed)", linewidth=4, color='blue')

# Indicate the final value on the y-axis
final_train_acc = smoothed_train_acc[-1]
final_val_acc = smoothed_val_acc[-1]
plt.axhline(y=final_train_acc, color='red', linestyle='--', linewidth=1)
plt.axhline(y=final_val_acc, color='blue', linestyle='--', linewidth=1)
x_right = epochs_range[-1]

plt.text(x_right, final_train_acc, f'{final_train_acc:.2f}%', color='red', va='center', ha='left',
         bbox=dict(facecolor='white', alpha=1, edgecolor='none'))
plt.text(x_right, final_val_acc, f'{final_val_acc:.2f}%', color='blue', va='center', ha='left',
         bbox=dict(facecolor='white', alpha=1, edgecolor='none'))

plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy (Smoothed)")
plt.legend()
plt.grid()
# plt.savefig("image9_perceptron_accuracy.png", dpi=600)
plt.show()