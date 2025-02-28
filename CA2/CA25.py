import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Define the Target Function
# -------------------------------
def target_function(x):
    return 1.2 * np.sin(np.pi * x) - np.cos(2.4 * np.pi * x)

# -------------------------------
# Step 2: Generate Training and Test Data
# -------------------------------
x_train = np.arange(-1.6, 1.6 + 0.05, 0.05)
y_train = target_function(x_train)

x_test = np.arange(-1.6, 1.6 + 0.01, 0.01)
y_test = target_function(x_test)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# -------------------------------
# Step 3: Define the MLP Model and Training Function
# -------------------------------
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

# -------------------------------
# Step 4: Define Hidden Neurons for Experiments
# -------------------------------
hidden_neurons = [1, 2, 4, 5, 10, 100]

# -------------------------------
# Step 5: Train and Plot MLP Results for Different Methods
# -------------------------------
def train_and_plot(method, optimizer_type, save_filename=None):
    results = {}
    plt.figure(figsize=(15, 10))

    for i, n in enumerate(hidden_neurons):
        model, y_pred = train_mlp(n, optimizer_type=optimizer_type)
        results[n] = y_pred

        plt.subplot(2, 3, i + 1)
        plt.plot(x_test, y_test, label="True Function", linestyle="dashed", linewidth=2)
        plt.plot(x_test, y_pred, label=f"MLP {n}-hidden", alpha=0.8, linewidth=2, color='red')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"MLP with {n} hidden neurons ({method})")
        plt.legend()

    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename, dpi=600)
    plt.show()

    return results

# Train using Backpropagation (BP)
# train_and_plot("BP", optimizer_type="adam", save_filename="image5_mlp_bp.png")

# # Train using trainlm (Levenberg-Marquardt)
# train_and_plot("trainlm", optimizer_type="lbfgs", save_filename="image6_mlp_trainlm.png")

# # Train using trainbr (Bayesian Regularization)
# train_and_plot("trainbr", optimizer_type="bayesian", save_filename="image7_mlp_trainbr.png")

# -------------------------------
# Step 6: Evaluate Out-of-Domain Predictions
# -------------------------------
x_out_of_domain = torch.tensor([[-3.0], [3.0]], dtype=torch.float32)

out_of_domain_results = {
    "BP": {},
    "trainlm": {},
    "trainbr": {}
}

for n in hidden_neurons:
    model, _ = train_mlp(n, optimizer_type="adam")
    out_of_domain_results["BP"][n] = model(x_out_of_domain).detach().numpy()

    model, _ = train_mlp(n, optimizer_type="lbfgs")
    out_of_domain_results["trainlm"][n] = model(x_out_of_domain).detach().numpy()

    model, _ = train_mlp(n, optimizer_type="bayesian", regularization=True)
    out_of_domain_results["trainbr"][n] = model(x_out_of_domain).detach().numpy()

# -------------------------------
# Step 7: Print Out-of-Domain Predictions
# -------------------------------
print("\nOut-of-domain predictions for x = -3 and x = 3:\n")
for n in hidden_neurons:
    print(f"MLP {n} hidden neurons - BP: {out_of_domain_results['BP'][n].flatten()}, "
          f"trainlm: {out_of_domain_results['trainlm'][n].flatten()}, "
          f"trainbr: {out_of_domain_results['trainbr'][n].flatten()}")
