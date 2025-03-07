import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader, TensorDataset

def load_images(folder_path, label):
    images, labels = [], []
    for i in range(500):  # 500 images per folder
        img_path = os.path.join(folder_path, f"{i:03d}.jpg")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = img.flatten() / 255.0  # Normalize and flatten to 1024
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Load datasets
animal_images, animal_labels = load_images(r"group_1/dog", label=1)
object_images, object_labels = load_images(r"group_1/automobile", label=0)

# Combine datasets
X = np.vstack((animal_images, object_images))
Y = np.hstack((animal_labels, object_labels))

# Shuffle the dataset randomly
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, Y = X[indices], Y[indices]

# Compute global mean and variance
global_mean = np.mean(X, axis=0)
global_variance = np.var(X, axis=0)

# Normalize dataset
X_normalized = (X - global_mean) / (np.sqrt(global_variance) + 1e-8)

# Ensure correct shape for input tensors
X_train = torch.tensor(X_normalized[:900], dtype=torch.float32)
Y_train = torch.tensor(Y[:900], dtype=torch.float32).unsqueeze(1)  # Add extra dimension

X_val = torch.tensor(X_normalized[900:], dtype=torch.float32)
Y_val = torch.tensor(Y[900:], dtype=torch.float32).unsqueeze(1)  # Add extra dimension

# Ensure shapes match
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")

# Create DataLoaders for batch processing
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

# -------------------------------
# Step 2: Define MLP Model
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_size=1024, hidden_size1=256, hidden_size2=128, hidden_size3=64):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.hidden3 = nn.Linear(hidden_size2, hidden_size3)
        self.output = nn.Linear(hidden_size3, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize model, loss, and optimizer
model = MLP()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Step 3: Train MLP with Batch Mode
# -------------------------------
num_epochs = 50
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Validation Phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            Y_pred = model(X_batch)
            val_loss += criterion(Y_pred, Y_batch).item()

    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

# -------------------------------
# Step 4: Evaluate Performance
# -------------------------------
def evaluate_accuracy(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            Y_pred = model(X_batch).round()
            correct += (Y_pred == Y_batch).sum().item()
            total += Y_batch.size(0)
    return 100 * correct / total

train_accuracy = evaluate_accuracy(train_loader)
val_accuracy = evaluate_accuracy(val_loader)

print(f"Final Training Accuracy: {train_accuracy:.2f}%")
print(f"Final Validation Accuracy: {val_accuracy:.2f}%")

import matplotlib.pyplot as plt
# -------------------------------
# Step 4: Plot Accuracy Over Epochs
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_accuracy, label="Training Accuracy", marker='o', color='red')
plt.plot(range(1, num_epochs + 1), val_accuracy, label="Validation Accuracy", marker='s', color='blue')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy Over Epochs")
plt.legend()
plt.grid()
plt.show()
