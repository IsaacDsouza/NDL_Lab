import numpy as np

class ErrorNeuron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()

    def activate(self, inputs):
        """Calculate output using sigmoid activation."""
        return 1 / (1 + np.exp(-np.dot(inputs, self.weights) - self.bias))

    def train(self, inputs, target, lr=0.1):
        """Update weights and bias using error correction."""
        output = self.activate(inputs)
        error = target - output
        self.weights += lr * error * inputs
        self.bias += lr * error


# Example usage
X_train = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
y_train = np.array([0, 1, 1, 0])

neuron = ErrorNeuron(num_inputs=3)

# Train the neuron for 10,000 iterations
for _ in range(10000):
    idx = np.random.randint(len(X_train))
    neuron.train(X_train[idx], y_train[idx])

# Test the neuron
for inputs in X_train:
    print("Input:", inputs, "Output:", neuron.activate(inputs))
