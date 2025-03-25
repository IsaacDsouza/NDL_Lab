import numpy as np

class MemoryBasedNeuron:
    def __init__(self):
        self.memory = []

    def train(self, data, labels):
        """Store training data in memory."""
        self.memory = list(zip(data, labels))

    def predict(self, sample, k=3):
        """Predict output using K-nearest neighbors."""
        distances = [(np.linalg.norm(sample - x), y) for x, y in self.memory]
        neighbors = sorted(distances, key=lambda d: d[0])[:k]
        return max(set([n[1] for n in neighbors]), key=[n[1] for n in neighbors].count)


# Example usage
data = np.array([[1, 2], [2, 3], [3, 3], [6, 7], [7, 8]])
labels = np.array([0, 0, 0, 1, 1])

neuron = MemoryBasedNeuron()
neuron.train(data, labels)

test_sample = np.array([4, 4])
prediction = neuron.predict(test_sample)
print("Predicted Label:", prediction)
