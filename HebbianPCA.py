import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate data
data = np.random.multivariate_normal([0, 0], [[3, 2], [2, 2]], 1000)

# Hebbian neuron model
class HebbianNeuron:
    def __init__(self, input_dim, lr=0.01):
        self.weights = np.random.randn(input_dim)
        self.lr = lr

    def train(self, data, epochs=10):
        for _ in range(epochs):
            for x in data:
                y = np.dot(self.weights, x)
                self.weights += self.lr * y * x

# Train neuron
neuron = HebbianNeuron(input_dim=2)
neuron.train(data)

# Extract principal component
pca_component = PCA(n_components=1).fit(data).components_[0]

# Normalize weights and component
w_norm = neuron.weights / np.linalg.norm(neuron.weights)
pca_norm = pca_component / np.linalg.norm(pca_component)

# Output and visualization
print("Neuron Weights:", w_norm)
print("PCA Component:", pca_norm)

plt.scatter(data[:, 0], data[:, 1], alpha=0.3, label="Data")
plt.quiver(0, 0, w_norm[0], w_norm[1], color='r', scale=3, label="Hebbian")
plt.quiver(0, 0, pca_norm[0], pca_norm[1], color='g', scale=3, label="PCA")
plt.legend(), plt.title("Hebbian vs PCA")
plt.axis('equal'), plt.grid(), plt.savefig("HebbianPCA.png")
