import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Define activation functions
def logistic(v):
    return 1 / (1 + np.exp(-v))

def gaussian(v):
    return np.exp(-((v - 5) / 2) ** 2)  # Using m = 2 as assumed

def softsign(v):
    return v / (1 + np.abs(v))

# Generate random data for binary classification
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a meshgrid for visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Train Logistic Regression (Hyperplane Decision Boundary)
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_boundary = logistic_model.predict_proba(grid_points)[:, 1].reshape(xx.shape)

# Train SVM with RBF Kernel for Gaussian Activation (Curved Decision Boundary)
gaussian_model = SVC(kernel='rbf', probability=True)
gaussian_model.fit(X_train, y_train)
gaussian_boundary = gaussian_model.predict_proba(grid_points)[:, 1].reshape(xx.shape)

# Train Logistic Regression for Softsign (Hyperplane Decision Boundary)
softsign_model = LogisticRegression()
softsign_model.fit(X_train, y_train)
softsign_boundary = softsign_model.predict_proba(grid_points)[:, 1].reshape(xx.shape)

# Plot decision boundaries with actual activation functions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Logistic Activation (Hyperplane)
axes[0].contourf(xx, yy, logistic_boundary, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'orange'])
axes[0].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
axes[0].set_title("Logistic Function - Hyperplane Boundary")

# Gaussian Activation (Curved Boundary via RBF Kernel)
axes[1].contourf(xx, yy, gaussian_boundary, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'orange'])
axes[1].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
axes[1].set_title("Gaussian Function - Curved Boundary")

# Softsign Activation (Hyperplane)
axes[2].contourf(xx, yy, softsign_boundary, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'orange'])
axes[2].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
axes[2].set_title("Softsign Function - Hyperplane Boundary")

plt.tight_layout()
plt.savefig("activation_functions.png", dpi=600)
plt.show()
