import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# Step 1: Load and Preprocess Data
# -------------------------------
def load_images(folder_path, label):
    images, labels = [], []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize for MobileNetV2
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])

    for i in range(500):  
        img_path = os.path.join(folder_path, f"{i:03d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip missing images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = transform(img)
        images.append(img)
        labels.append(label)

    return torch.stack(images), torch.tensor(labels, dtype=torch.float32)

# Load datasets
animal_images, animal_labels = load_images(r"CA2/group_1/dog", label=1)
object_images, object_labels = load_images(r"CA2/group_1/automobile", label=0)

# Combine datasets
X = torch.cat((animal_images, object_images), dim=0)
Y = torch.cat((animal_labels, object_labels), dim=0)

# Shuffle dataset
indices = torch.randperm(X.shape[0])
X, Y = X[indices], Y[indices]

# Split dataset (90:10 train-val split)
split_index = int(0.9 * X.shape[0])
X_train, Y_train = X[:split_index], Y[:split_index]
X_val, Y_val = X[split_index:], Y[split_index:]

# Convert labels to shape [N, 1] for Binary Classification
Y_train, Y_val = Y_train.unsqueeze(1), Y_val.unsqueeze(1)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

# -------------------------------
# Step 2: Load MobileNetV2 as Feature Extractor
# -------------------------------
mobilenet = models.mobilenet_v2(pretrained=True)
feature_extractor = mobilenet.features  # Extract convolutional layers
input_features = mobilenet.last_channel  # 1280 output features

for param in feature_extractor.parameters():
    param.requires_grad = False  # Freeze convolutional layers

# -------------------------------
# Step 3: Define Optimized MLP Classifier
# -------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)  # More dropout in early layers
        self.dropout2 = nn.Dropout(0.3)  # Less dropout in deeper layers
        self.sigmoid = nn.Sigmoid()
        
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout1(x)
        x = self.relu(self.batchnorm3(self.fc3(x)))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc4(x))
        return x

# -------------------------------
# Step 4: Train MLP with MobileNetV2 Features
# -------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
feature_extractor = feature_extractor.to(device)
mlp_classifier = MLPClassifier(input_features).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(mlp_classifier.parameters(), lr=0.001, weight_decay=2e-3)  # Stronger L2 regularization
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Early stopping setup
num_epochs = 100
patience = 15
best_val_loss = float("inf")
patience_counter = 0

train_losses, val_losses, train_acc, val_acc = [], [], [], []

for epoch in range(num_epochs):
    mlp_classifier.train()
    epoch_loss = 0
    correct, total = 0, 0

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Extract features from MobileNetV2
        with torch.no_grad():
            features = feature_extractor(X_batch)
            features = features.mean([2, 3])  # Global average pooling

        optimizer.zero_grad()
        Y_pred = mlp_classifier(features)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(mlp_classifier.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_loss += loss.item()
        correct += ((Y_pred > 0.5).float() == Y_batch).sum().item()
        total += Y_batch.size(0)

    train_losses.append(epoch_loss / len(train_loader))
    train_acc.append(100 * correct / total)

    # Validation phase
    mlp_classifier.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            features = feature_extractor(X_batch)
            features = features.mean([2, 3])  # Global average pooling
            Y_pred = mlp_classifier(features)

            val_loss += criterion(Y_pred, Y_batch).item()
            correct += ((Y_pred > 0.5).float() == Y_batch).sum().item()
            total += Y_batch.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_acc.append(100 * correct / total)

    print(f"Epoch {epoch+1}: Train Acc = {train_acc[-1]:.2f}%, Val Acc = {val_acc[-1]:.2f}%")

    scheduler.step(val_losses[-1])  # Adaptive LR

    # Early stopping check
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# -------------------------------
# Step 5: Smooth and Plot Accuracy Trends
# -------------------------------
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Define smoothing window
window_size = 5

# Smooth accuracy values
smoothed_train_acc = moving_average(train_acc, window_size)
smoothed_val_acc = moving_average(val_acc, window_size)

# Adjust x-axis to match smoothed data
epochs_range = range(len(smoothed_train_acc))

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, smoothed_train_acc, label="Train Accuracy (Smoothed)", color='red', linewidth=3)
plt.plot(epochs_range, smoothed_val_acc, label="Validation Accuracy (Smoothed)", color='blue', linewidth=3)

# Annotate final values
final_train_acc = smoothed_train_acc[-1]
final_val_acc = smoothed_val_acc[-1]
plt.axhline(y=final_train_acc, color='red', linestyle='--', linewidth=1)
plt.axhline(y=final_val_acc, color='blue', linestyle='--', linewidth=1)

# Display final accuracy as text on the graph
plt.text(epochs_range[-1], final_train_acc, f'{final_train_acc:.2f}%', color='red', va='center', ha='left',
         bbox=dict(facecolor='white', alpha=1, edgecolor='none'))
plt.text(epochs_range[-1], final_val_acc, f'{final_val_acc:.2f}%', color='blue', va='center', ha='left',
         bbox=dict(facecolor='white', alpha=1, edgecolor='none'))

plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Smoothed Training vs Validation Accuracy with MobileNetV2")
plt.legend()
plt.grid()
plt.savefig("images14_mobilenetv2.png", dpi=600)
plt.show()

