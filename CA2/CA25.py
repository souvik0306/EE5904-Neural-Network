import matplotlib.pyplot as plt

# Define hidden neurons used in the experiment
hidden_neurons = [1, 2, 4, 5, 10, 100]

# Define values for each method at x = -3 and x = 3
bp_xneg3 = [0.18919289, 2.1454768, 1.3331243, -1.879192, -3.7878022, -4.571954]
trainlm_xneg3 = [0.7532759, 1.3004378, 1.4587393, 2.6031017, 3.7843096, -1.1215659]
trainbr_xneg3 = [0.6105115, 1.5679431, 0.67763, 1.7362639, 4.6818137, 5.0687246]

bp_x3 = [-2.8801906, -3.7305717, -2.7517562, -0.71682286, -3.5645664, -4.2341743]
trainlm_x3 = [-0.7138791, -2.010186, -2.1604395, -0.91965103, -1.0808313, 2.2052255]
trainbr_x3 = [-0.5151646, 0.14981228, -0.62427133, -1.4997219, -4.6989074, -5.0716743]

# Create subplots for -3 and +3 predictions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot for x = -3
axes[0].plot(hidden_neurons, bp_xneg3, 'ro-', label="BP (y=-3)")
axes[0].plot(hidden_neurons, trainlm_xneg3, 'bo-', label="trainlm (y=-3)")
axes[0].plot(hidden_neurons, trainbr_xneg3, 'go-', label="trainbr (y=-3)")
axes[0].axhline(y=-3, color='black', linestyle='dashed', linewidth=2, label="True Value (y=-3)")
axes[0].set_xlabel("Number of Hidden Neurons")
axes[0].set_ylabel("Prediction at y=-3")
axes[0].set_title("Out-of-Domain Predictions at y=-3")
axes[0].legend()

# Subplot for x = 3
axes[1].plot(hidden_neurons, bp_x3, 'r--o', label="BP (y=3)")
axes[1].plot(hidden_neurons, trainlm_x3, 'b--o', label="trainlm (y=3)")
axes[1].plot(hidden_neurons, trainbr_x3, 'g--o', label="trainbr (y=3)")
axes[1].axhline(y=3, color='black', linestyle='dashed', linewidth=2, label="True Value (y=3)")
axes[1].set_xlabel("Number of Hidden Neurons")
axes[1].set_ylabel("Prediction at y=3")
axes[1].set_title("Out-of-Domain Predictions at y=3")
axes[1].legend()

plt.tight_layout()
plt.savefig("image8_mlp_out_of_domain.png", dpi=600)
plt.show()
