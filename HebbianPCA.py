import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate data
inputs = np.random.multivariate_normal([0, 0], [[3, 2], [2, 2]], 1000)

# Hebbian neuron model
class HebbianNeuron:
    def __init__(self, num_inputs, lr=0.01):
        self.weights = np.random.randn(num_inputs)
        self.lr = lr

    def train(self, inputs, epochs=10):
        for _ in range(epochs):
            for x in inputs:
                y = np.dot(self.weights, x)
                self.weights += self.lr * y * x

# Train neuron
neuron = HebbianNeuron(num_inputs=2)
neuron.train(inputs)

# Extract principal component
pca_component = PCA(n_components=1).fit(inputs).components_[0]

# Normalize weights and component
w_norm = neuron.weights / np.linalg.norm(neuron.weights)
pca_norm = pca_component / np.linalg.norm(pca_component)

# Output and visualization
print("Neuron Weights:", w_norm)
print("PCA Component:", pca_norm)

plt.scatter(inputs[:, 0], inputs[:, 1], alpha=0.3, label="Data")
plt.quiver(0, 0, w_norm[0], w_norm[1], color='r', scale=3, label="Hebbian")
plt.quiver(0, 0, pca_norm[0], pca_norm[1], color='g', scale=3, label="PCA")
plt.legend(), plt.title("Hebbian vs PCA")
plt.axis('equal'), plt.grid(), plt.savefig("HebbianPCA.png")