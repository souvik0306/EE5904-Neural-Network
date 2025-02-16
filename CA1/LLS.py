# Re-import necessary libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt

# Given data points
X = np.array([0, 0.8, 1.6, 3, 4.0, 5.0])
y = np.array([0.5, 1, 4, 5, 6, 8])

# Part (a): Solve for w and b using the Linear Least Squares (LLS) method
X_matrix = np.vstack([X, np.ones(len(X))]).T  # Add bias term
w_lls, b_lls = np.linalg.lstsq(X_matrix, y, rcond=None)[0]  # Solve using LLS

# Generate predictions
X_fit = np.linspace(min(X), max(X), 100)
y_fit_lls = w_lls * X_fit + b_lls

# Plot LLS fitting
plt.figure(figsize=(6, 5))
plt.scatter(X, y, color='red', label="Data Points")
plt.plot(X_fit, y_fit_lls, label=f"LLS Fit: y = {w_lls:.4f}x + {b_lls:.4f}", color='blue')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Fit using Least Squares (LLS) Method")
plt.legend()
plt.grid(True)
plt.savefig("images4_LLS.png", dpi=600)
plt.show()

# Print LLS results
print(f"Solution using LLS method: w = {w_lls:.4f}, b = {b_lls:.4f}")
