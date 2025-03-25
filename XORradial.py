import numpy as np
import matplotlib.pyplot as plt

# Define inputs and outputs for XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 0])

# Gaussian activation function
def gaussian(v, w):  
    return np.exp(-np.linalg.norm(v - w)**2)

# Define centers for RBF neurons
w1, w2 = np.array([1, 1]), np.array([0, 0])

# Calculate activations
f1, f2 = [gaussian(i, w1) for i in inputs], [gaussian(i, w2) for i in inputs]

# Plot hidden neuron activations
plt.scatter(f1[:2], f2[:2], marker="x", label="Class 0")
plt.scatter(f1[2:], f2[2:], marker="o", label="Class 1")
plt.plot(np.linspace(0, 1, 10), -np.linspace(0, 1, 10) + 1, label="Decision Boundary")
plt.xlabel("Hidden Function 1"), plt.ylabel("Hidden Function 2")
plt.legend()
plt.savefig("XOR_RBF_plot.png")  # Save the plot to file

