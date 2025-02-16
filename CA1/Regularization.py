import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Create synthetic dataset
N = 30                   # Number of samples
X_true = np.linspace(0, 10, N)  # True x values
noise = np.random.normal(0, 2, N)  # Gaussian noise
slope_true = 2.0
intercept_true = 3.0
y_true = slope_true * X_true + intercept_true

# Observed data with noise
y_observed = y_true + noise

# Matrix form: X matrix of shape (N, 2) for [x, 1]
X_mat = np.column_stack((X_true, np.ones(N)))

# Define weights based on x
weights = 1.0 + X_true / 10.0  # e.g., bigger weights for larger x
R = np.diag(weights)           # Diagonal matrix of weights

def solve_weighted_ridge(X, y, R, lam):
    """
    Solve the regularized weighted least squares problem:
    w* = (X^T R X + λI)^(-1) X^T R y
    """
    # X: (N,2), y: (N,), R: (N,N), lam: float
    A = X.T @ R @ X + lam * np.eye(X.shape[1])
    b = X.T @ R @ y
    w_opt = np.linalg.inv(A) @ b
    return w_opt

# Solve for unregularized weighted least squares
w_unreg = solve_weighted_ridge(X_mat, y_observed, R, lam=0.0)

# Solve for moderately regularized weighted least squares
w_reg = solve_weighted_ridge(X_mat, y_observed, R, lam=5.0)

print("Unregularized Weighted LS:")
print(f"  slope = {w_unreg[0]:.3f}, intercept = {w_unreg[1]:.3f}")

print("Regularized Weighted LS (lambda=5):")
print(f"  slope = {w_reg[0]:.3f}, intercept = {w_reg[1]:.3f}")

# Generate a smooth range for plotting
x_plot = np.linspace(0, 10, 100)
X_plot_mat = np.column_stack((x_plot, np.ones_like(x_plot)))

# Predictions for each line
y_unreg = X_plot_mat @ w_unreg
y_reg   = X_plot_mat @ w_reg
# Plot
plt.figure(figsize=(11,6))
plt.scatter(X_true, y_observed, label="Data (Weighted)", color='red')

plt.plot(x_plot, y_unreg, label=f"Unregularized Weighted LS", color='blue')
plt.plot(x_plot, y_reg,   label=f"Regularized Weighted LS (λ=5)", 
         color='green', linestyle='--')

# Fill the area between the two lines to show the difference
plt.fill_between(x_plot, y_unreg, y_reg, color='gray', alpha=0.3, label='Difference Area')
# Add an arrow pointing to the difference area
plt.annotate(
    'Outliers contributed to the Unregularised hyperplane shifting away', 
    xy=(2, (y_unreg[20] + y_reg[30]) / 2), 
    xytext=(4, (y_unreg[20] + y_unreg[20]) / 2 - 5),
    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=6),
    fontsize=10, color='purple',
    wrap=True
)

# Annotate outliers
for i in range(N):
    if abs(noise[i]) > 2:  # Assuming noise > 2 as outliers
        # Adjust the y position of the annotation to avoid overlapping with the line
        y_offset = 10 if y_observed[i] < (slope_true * X_true[i] + intercept_true) else -10
        plt.annotate('Outlier', (X_true[i], y_observed[i]), textcoords="offset points", xytext=(0, y_offset), ha='center', color='purple')

plt.xlabel("X")
plt.ylabel("y")
plt.title("Regularized v/s Unregularized Weighted Least Squares")
plt.legend()
plt.grid(True)
plt.savefig("images4_WeightedRidge.png", dpi=600)
plt.show()

# Annotate outliers
for i in range(N):
    if abs(noise[i]) > 2:  # Assuming noise > 2 as outliers
        plt.annotate('Outlier', (X_true[i], y_observed[i]), textcoords="offset points", xytext=(0,10), ha='center', color='purple')
