import numpy as np

# Perceptron class
class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        self.weights = np.random.rand(input_size + 1)  # +1 for bias
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights[1:]) + self.weights[0])

    def train(self, X, y):
        for _ in range(self.epochs):
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction
                self.weights[1:] += self.lr * error * inputs
                self.weights[0] += self.lr * error

# Define logic gates and their truth tables
logic_gates = {
    "AND": ([0, 0, 1, 1], [0, 1, 0, 1]),
    "OR": ([0, 0, 1, 1], [0, 1, 1, 1]),
    "NAND": ([0, 0, 1, 1], [1, 1, 1, 0]),
    "NOR": ([0, 0, 1, 1], [1, 0, 0, 0]),
    "XNOR": ([0, 0, 1, 1], [1, 0, 0, 1]),
    "XOR": ([0, 0, 1, 1], [0, 1, 1, 0]),
    "NOT": ([0, 1], [1, 0]),
}

# Train and test each gate
for gate, (inputs, outputs) in logic_gates.items():
    X = np.array([list(bin(i)[2:].zfill(2)) for i in range(len(inputs))], dtype=int)
    y = np.array(outputs)
    if gate == "NOT":  # NOT gate only needs 1 input
        X = X[:, :1]

    perceptron = Perceptron(input_size=X.shape[1])
    perceptron.train(X, y)

    print(f"\n{gate} Gate:")
    for inp in X:
        print(f"Input: {inp} -> Output: {perceptron.predict(inp)}")
