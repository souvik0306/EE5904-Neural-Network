# Full Python Code for Gradient Descent on the Rosenbrock Function

import numpy as np
import matplotlib.pyplot as plt

# Define the Rosenbrock function
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Define gradient of Rosenbrock function
def grad_rosenbrock(x, y):
    df_dx = -2 * (1 - x) - 400 * x * (y - x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

# Gradient Descent Implementation
def gradient_descent(start, eta=0.001, tol=1e-6, max_iter=10000):
    x, y = start
    trajectory = [(x, y)]
    function_values = [rosenbrock(x, y)]
    
    for i in range(max_iter):
        grad = grad_rosenbrock(x, y)
        x, y = x - eta * grad[0], y - eta * grad[1]
        trajectory.append((x, y))
        function_values.append(rosenbrock(x, y))

        if np.linalg.norm(grad) < tol:
            break

    return np.array(trajectory), np.array(function_values), i+1

# Generate a random starting point in (-1,1) for x and y
np.random.seed(42)
start = np.random.uniform(-1, 1, 2)

# Run gradient descent with eta = 0.001
traj_gd, values_gd, iter_gd = gradient_descent(start, eta=0.001)

# Run gradient descent with eta = 1.0 to test divergence
traj_gd_large_eta, values_gd_large_eta, iter_gd_large_eta = gradient_descent(start, eta=1.0)

# Plotting results
plt.figure(figsize=(12, 5))

# 2D Trajectory Plot
plt.subplot(1, 2, 1)
plt.plot(traj_gd[:, 0], traj_gd[:, 1], marker='o', markersize=2, label="Gradient Descent (η = 0.001)", color='blue', markerfacecolor='yellow', markeredgewidth=3.5)
plt.scatter(1, 1, color='red', marker='x', s=100, label="Global Minimum (1,1)", linewidths=3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory of (x, y) in 2D Space")
plt.legend()

# Function value convergence plot
plt.subplot(1, 2, 2)
plt.plot(values_gd, label="Function Value (η = 0.001)", color='blue')
plt.xlabel("Iterations")
plt.ylabel("f(x, y)")
plt.title("Function Value Convergence")
plt.legend()

plt.tight_layout()
plt.savefig("image2_rosenbrock.png", dpi=600)
plt.show()

# Check if eta = 1.0 leads to divergence
if values_gd_large_eta[-1] > 1e6:  # Arbitrary large value indicating divergence
    divergence_message = "With η = 1.0, the function value explodes, indicating divergence."
else:
    divergence_message = "With η = 1.0, the function value still converges, but may be unstable."

# Output results
iter_gd, divergence_message
