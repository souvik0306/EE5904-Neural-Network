import numpy as np
import matplotlib.pyplot as plt

# Define v values
v = np.linspace(-5, 5, 400)

# Define activation functions
logistic = 1 / (1 + np.exp(-v))
gaussian = np.exp(-((v - 2)** 2)/2)  # Assuming m = 2
softsign = v / (1 + np.abs(v))

# Plot activation functions
plt.figure(figsize=(12, 4))

# Logistic Function
plt.subplot(1, 3, 1)
plt.plot(v, logistic, label=r'$\phi(v) = \frac{1}{1+e^{-v}}$', color='b', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
plt.xlabel("v")
plt.ylabel("φ(v)")
plt.title("Logistic Function")
plt.legend()
plt.grid()

# Gaussian Function
plt.subplot(1, 3, 2)
plt.plot(v, gaussian, label=r'$\phi(v) = e^\frac{-(v-2)^2}{2}$', color='g', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
plt.xlabel("v")
plt.ylabel("φ(v)")
plt.title("Gaussian Function")
plt.legend()
plt.grid()

# Softsign Function
plt.subplot(1, 3, 3)
plt.plot(v, softsign, label=r'$\phi(v) = \frac{v}{1+|v|}$', color='m', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
plt.xlabel("v")
plt.ylabel("φ(v)")
plt.title("Softsign Function")
plt.legend()
plt.grid()

# Show the plots
plt.tight_layout()
plt.savefig("activation_functions2.png", dpi=600)
plt.show()
