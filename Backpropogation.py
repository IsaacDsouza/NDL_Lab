import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001, time_steps=5):
        self.hidden_dim, self.time_steps, self.lr = hidden_dim, time_steps, lr
        # Initialize weights & biases
        self.Wxh, self.Whh, self.Why = [np.random.randn(*shape) * 0.01 for shape in [
            (hidden_dim, input_dim), (hidden_dim, hidden_dim), (output_dim, hidden_dim)]]
        self.bh, self.by = np.zeros((hidden_dim, 1)), np.zeros((output_dim, 1))

    def forward(self, inputs):
        h, self.h_states, self.outputs = np.zeros((self.hidden_dim, 1)), {-1: np.zeros((self.hidden_dim, 1))}, {}
        for t in range(self.time_steps):
            h = np.tanh(np.dot(self.Wxh, inputs[t]) + np.dot(self.Whh, self.h_states[t-1]) + self.bh)
            self.h_states[t], self.outputs[t] = h, np.dot(self.Why, h) + self.by
        return self.outputs

    def backward(self, inputs, targets):
        dWxh, dWhh, dWhy, dbh, dby = [np.zeros_like(param) for param in [self.Wxh, self.Whh, self.Why, self.bh, self.by]]
        dh_next = np.zeros((self.hidden_dim, 1))

        for t in reversed(range(self.time_steps)):
            dy = self.outputs[t] - targets[t]
            dWhy += np.dot(dy, self.h_states[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - self.h_states[t]**2) * dh
            dbh += dh_raw
            dWxh += np.dot(dh_raw, inputs[t].T)
            dWhh += np.dot(dh_raw, self.h_states[t-1].T)
            dh_next = np.dot(self.Whh.T, dh_raw)

        # Clip and update weights
        for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], [dWxh, dWhh, dWhy, dbh, dby]):
            np.clip(dparam, -5, 5, out=dparam)
            param -= self.lr * dparam

    def train(self, data, labels, epochs=100):
        for epoch in range(epochs):
            loss = sum(np.sum((self.forward(x)[self.time_steps-1] - y[self.time_steps-1])**2) / 2 
                       for x, y in zip(data, labels))
            for x, y in zip(data, labels):
                self.backward(x, y)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example usage
if __name__ == "__main__":
    time_steps, input_dim, hidden_dim, output_dim = 5, 3, 4, 2
    data = [np.random.randn(time_steps, input_dim, 1) * 0.1 for _ in range(100)]
    labels = [np.random.randn(time_steps, output_dim, 1) * 0.1 for _ in range(100)]
    rnn = RNN(input_dim, hidden_dim, output_dim)
    rnn.train(data, labels, epochs=50)
