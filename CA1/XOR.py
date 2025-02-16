import numpy as np
import matplotlib.pyplot as plt

# Define XOR input and output
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Plot XOR data points
plt.figure(figsize=(6, 4))
for i, (x1, x2) in enumerate(X_xor):
    plt.scatter(x1, x2, color='red' if y_xor[i] == 1 else 'blue', s=100, linewidths=4, label=f'Class {y_xor[i]}' if i < 2 else "")

# Draw x and y axes
plt.axhline(0, color='black', linewidth=.5)
plt.axvline(0, color='black', linewidth=.5)

# Labels and title
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("XOR Problem - Non-Linearly Separable Data")
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.grid(True)
plt.legend()
plt.savefig("images3.png", dpi=600)
plt.show()
