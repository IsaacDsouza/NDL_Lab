import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, input_dim, grid_size, lr=0.1, epochs=1000):
        self.weights = np.random.rand(*grid_size, input_dim)
        self.lr, self.epochs = lr, epochs
        self.radius = max(grid_size) / 2

    def find_bmu(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=2)
        return tuple(np.unravel_index(np.argmin(distances), distances.shape))

    def update_weights(self, sample, bmu):
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                dist = np.linalg.norm(np.array([i, j]) - np.array(bmu))
                if dist <= self.radius:
                    influence = np.exp(-dist**2 / (2 * self.radius**2))
                    self.weights[i, j] += self.lr * influence * (sample - self.weights[i, j])

    def train(self, data):
        for _ in range(self.epochs):
            for sample in data:
                bmu = self.find_bmu(sample)
                self.update_weights(sample, bmu)
            self.lr *= 0.995
            self.radius *= 0.995

    def visualize(self, filename="som_output.png"):
        plt.imshow(self.weights.reshape(*self.weights.shape[:2], -1))
        plt.title("Self-Organizing Map")
        plt.savefig(filename)  # Save the figure as an image file
        plt.close()


# Example usage
data = np.random.rand(100, 3)  # 100 samples, 3 features
som = SOM(input_dim=3, grid_size=(10, 10))
som.train(data)
som.visualize()
