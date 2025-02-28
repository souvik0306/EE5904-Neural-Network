# Re-import required libraries after execution state reset
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

# Define Hessian of Rosenbrock function
def hessian_rosenbrock(x, y):
    d2f_dx2 = 2 - 400 * (y - 3*x**2)
    d2f_dxdy = -400 * x
    d2f_dy2 = 200
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

# Newton's Method Implementation
def newtons_method(start, tol=1e-6, max_iter=100):
    x, y = start
    trajectory = [(x, y)]
    function_values = [rosenbrock(x, y)]
    
    for i in range(max_iter):
        grad = grad_rosenbrock(x, y)
        hess = hessian_rosenbrock(x, y)
        hess_inv = np.linalg.inv(hess)
        update = -hess_inv @ grad
        x, y = x + update[0], y + update[1]
        
        trajectory.append((x, y))
        function_values.append(rosenbrock(x, y))

        if np.linalg.norm(grad) < tol:
            break

    return np.array(trajectory), np.array(function_values), i+1

# Generate a random starting point in (-1,1) for x and y
np.random.seed(42)
start = np.random.uniform(-1, 1, 2)

# Run Newton's Method
traj_newton, values_newton, iter_newton = newtons_method(start)

# Plotting results
plt.figure(figsize=(12, 5))

# 2D Trajectory Plot
plt.subplot(1, 2, 1)
plt.plot(traj_newton[:, 0], traj_newton[:, 1], marker='s', markersize=5, label="Newton's Method", color='green', linewidth=3, markerfacecolor='yellow')
plt.scatter(1, 1, color='red', marker='x', s=100, label="Global Minimum (1,1)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory of (x, y) using Newton's Method")
plt.legend()

# Function value convergence plot
plt.subplot(1, 2, 2)
plt.plot(values_newton, label="Function Value (Newton's Method)", color='green',linewidth=4)
plt.xlabel("Iterations")
plt.ylabel("f(x, y)")
plt.title("Function Value Convergence using Newton's Method")
plt.legend()

plt.tight_layout()
plt.savefig("images3_newton_method.png", dpi=600)
plt.show()

# Output number of iterations taken
iter_newton
