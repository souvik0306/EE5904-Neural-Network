import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(seed=705)

# -------------------------------
# Step 1: Load and Normalize Data
# -------------------------------
def load_images(folder_path, label):
    images, labels = [], []
    for i in range(500):  # 500 images per class
        img_path = os.path.join(folder_path, f"{i:03d}.jpg")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = img.flatten() / 255.0  # Normalize (0-1 range)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Load both classes
animal_images, animal_labels = load_images(r"CA2/group_1/dog", label=1)
object_images, object_labels = load_images(r"CA2/group_1/automobile", label=0)

# Combine dataset
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

# Convert to PyTorch tensors & ensure correct shape
X_train = torch.tensor(X_normalized[:900], dtype=torch.float32)
Y_train = torch.tensor(Y[:900], dtype=torch.float32).unsqueeze(1)  # Shape [900, 1]

X_val = torch.tensor(X_normalized[900:], dtype=torch.float32)
Y_val = torch.tensor(Y[900:], dtype=torch.float32).unsqueeze(1)  # Shape [100, 1]

# Create DataLoaders for batch processing
batch_size = 128
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False, pin_memory=True)

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
        
        # Added BatchNorm & Dropout for stability
        self.batchnorm1 = nn.BatchNorm1d(hidden_size1)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size2)
        self.batchnorm3 = nn.BatchNorm1d(hidden_size3)
        self.dropout = nn.Dropout(0.5)  # 20% dropout

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.hidden1(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.hidden2(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm3(self.hidden3(x)))
        x = self.sigmoid(self.output(x))
        return x

# -------------------------------
# Step 3: Train MLP with Early Stopping & Accuracy Tracking
# -------------------------------

# Set device for Apple Silicon (MPS)
device = torch.device("cpu")
print(f"Using device: {device}")

# Initialize model, loss, and optimizer
model = MLP().to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-2)

num_epochs = 100
patience = 20  # Stop training if no improvement for 10 epochs
best_val_loss = float("inf")
patience_counter = 0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []  # Track accuracies

def evaluate_accuracy(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Y_pred = model(X_batch).round()
            correct += (Y_pred == Y_batch).sum().item()
            total += Y_batch.size(0)
    return 100 * correct / total

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
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
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Y_pred = model(X_batch)
            val_loss += criterion(Y_pred, Y_batch).item()

    val_losses.append(val_loss / len(val_loader))

    # Compute accuracy
    train_accuracy = evaluate_accuracy(train_loader)
    val_accuracy = evaluate_accuracy(val_loader)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
    print(f"Train Acc = {train_accuracy:.2f}%, Val Acc = {val_accuracy:.2f}%")

    # Early Stopping Logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")  # Save best model
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}. Best Validation Loss: {best_val_loss:.4f}")
        break

# Load best model
model.load_state_dict(torch.load("best_model.pth"))

# -------------------------------
# Step 4: Plot Smoothed Accuracy Curves
# -------------------------------

def moving_average(data, window_size=5):
    """Compute moving average to smooth the accuracy curves."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Apply smoothing
window_size = 5
smoothed_train_acc = moving_average(train_accuracies, window_size)
smoothed_val_acc = moving_average(val_accuracies, window_size)
epochs_range = range(1, len(smoothed_train_acc) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, smoothed_train_acc, label="Train Accuracy (Smoothed)", linewidth=4, color='red')
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
plt.title("Batch Smoothed Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.savefig("image11_percepton_accuracy.png", dpi=600)
plt.show()
