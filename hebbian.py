import numpy as np

class HebbianNeuron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)

    def activate(self, inputs):
        return np.dot(inputs, self.weights)
    
    def train(self, inputs):
        output = self.activate(inputs)
        self.weights += output*inputs
        """Hebbian learning rule: Δw = input * output."""


X_train = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])


neuron = HebbianNeuron(num_inputs=3)

for inputs in X_train:
    neuron.train(inputs)

for inputs in X_train:
    print("Input:", inputs, "Output:", neuron.activate(inputs))