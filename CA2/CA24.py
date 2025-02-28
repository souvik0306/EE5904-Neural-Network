import numpy as np
import torch

# Define the target function
def target_function(x):
    return 1.2 * np.sin(np.pi * x) - np.cos(2.4 * np.pi * x)

# Generate training and test data
x_train = np.arange(-1.6, 1.6 + 0.05, 0.05)
y_train = target_function(x_train)

x_test = np.arange(-1.6, 1.6 + 0.01, 0.01)
y_test = target_function(x_test)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

import torch.nn as nn
import torch.optim as optim

# Function to create and train an MLP
def train_mlp(n_hidden, optimizer_type="adam", epochs=2000, learning_rate=0.01, regularization=False):
    model = nn.Sequential(
        nn.Linear(1, n_hidden),
        nn.Tanh(),
        nn.Linear(n_hidden, 1)
    )
    
    criterion = nn.MSELoss()
    
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)
    elif optimizer_type == "bayesian":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01 if regularization else 0.0)

    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            y_pred = model(x_train_tensor)
            loss = criterion(y_pred, y_train_tensor)
            loss.backward()
            return loss
        
        if optimizer_type == "lbfgs":
            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            y_pred = model(x_train_tensor)
            loss = criterion(y_pred, y_train_tensor)
            loss.backward()
            optimizer.step()

    # Evaluate on test data
    y_test_pred = model(x_test_tensor).detach().numpy()

    return model, y_test_pred

import matplotlib.pyplot as plt

# Train different MLP architectures using Backpropagation
hidden_neurons = [1, 2, 3,4,6,7,8, 10, 50, 100]
# results_bp = {}

# plt.figure(figsize=(15, 12))
# for i, n in enumerate(hidden_neurons[:10]):  # Limit visualization to 10 cases
#     model, y_pred = train_mlp(n, optimizer_type="adam")
#     results_bp[n] = y_pred
    
#     plt.subplot(2, 3, i+1)
#     plt.plot(x_test, y_test, label="True Function", linestyle="dashed", linewidth=2)
#     plt.plot(x_test, y_pred, label=f"MLP {n}-hidden", alpha=0.8, linewidth=2, color='red')
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title(f"MLP with {n} hidden neurons (BP)")
#     plt.legend()

# plt.tight_layout()
# plt.savefig("image5_mlp_bp.png", dpi=600)
# plt.show()

# Train using trainlm (LBFGS optimizer equivalent)
# results_trainlm = {}

# plt.figure(figsize=(15, 12))
# for i, n in enumerate(hidden_neurons[:10]):
#     model, y_pred = train_mlp(n, optimizer_type="lbfgs")
#     results_trainlm[n] = y_pred
    
#     plt.subplot(2, 3, i+1)
#     plt.plot(x_test, y_test, label="True Function", linestyle="dashed", linewidth=2)
#     plt.plot(x_test, y_pred, label=f"MLP {n}-hidden", alpha=0.8, linewidth=2)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title(f"MLP with {n} hidden neurons (trainlm)")
#     plt.legend()

# plt.tight_layout()
# plt.savefig("image6_mlp_trainlm.png", dpi=600)
# plt.show()

# Train using trainbr (Bayesian Regularization)
results_trainbr = {}

plt.figure(figsize=(15, 12))
for i, n in enumerate(hidden_neurons[:10]):
    model, y_pred = train_mlp(n, optimizer_type="bayesian", regularization=True)
    results_trainbr[n] = y_pred
    
    plt.subplot(5, 2, i+1)
    plt.plot(x_test, y_test, label="True Function", linestyle="dashed", linewidth=2)
    plt.plot(x_test, y_pred, label=f"MLP {n}-hidden", alpha=0.8, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"MLP with {n} hidden neurons (trainbr)")
    plt.legend()

plt.tight_layout()
# plt.savefig("image7_mlp_trainbr.png", dpi=600)
plt.show()

# x_out_of_domain = torch.tensor([[-3.0], [3.0]], dtype=torch.float32)
# y_out_pred_bp = {}
# y_out_pred_trainlm = {}
# y_out_pred_trainbr = {}

# for n in hidden_neurons[:10]:
#     model, _ = train_mlp(n, optimizer_type="adam")
#     y_out_pred_bp[n] = model(x_out_of_domain).detach().numpy()

#     model, _ = train_mlp(n, optimizer_type="lbfgs")
#     y_out_pred_trainlm[n] = model(x_out_of_domain).detach().numpy()

#     model, _ = train_mlp(n, optimizer_type="bayesian", regularization=True)
#     y_out_pred_trainbr[n] = model(x_out_of_domain).detach().numpy()

# print("\nOut-of-domain predictions for x = -3 and x = 3:\n")
# for n in hidden_neurons[:10]:
#     print(f"MLP {n} hidden neurons - BP: {y_out_pred_bp[n].flatten()}, trainlm: {y_out_pred_trainlm[n].flatten()}, trainbr: {y_out_pred_trainbr[n].flatten()}")

# # Plotting the out-of-domain predictions
# plt.figure(figsize=(15, 12))
# for i, n in enumerate(hidden_neurons[:10]):
#     plt.subplot(5, 2, i+1)
#     plt.plot([-3, 3], y_out_pred_bp[n], label="BP", marker='o')
#     plt.plot([-3, 3], y_out_pred_trainlm[n], label="trainlm", marker='x')
#     plt.plot([-3, 3], y_out_pred_trainbr[n], label="trainbr", marker='s')
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title(f"Out-of-domain predictions with {n} hidden neurons")
#     plt.legend()

# plt.tight_layout()
# plt.show()