import numpy as np

# Define activation functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

# XOR dataset
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
outputs = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
np.random.seed(1)
input_size, hidden_size, output_size = 2, 2, 1
W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
learning_rate = 0.1

# Train the network
for epoch in range(10000):
    # Forward pass
    hidden_input = np.dot(inputs, W1)
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2)
    final_output = sigmoid(final_input)

    # Calculate error
    error = outputs - final_output
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error)):.4f}")

    # Backpropagation
    d_output = error * sigmoid_deriv(final_output)
    d_hidden = d_output.dot(W2.T) * sigmoid_deriv(hidden_output)

    # Update weights
    W2 += hidden_output.T.dot(d_output) * learning_rate
    W1 += inputs.T.dot(d_hidden) * learning_rate

# Test the XOR output
print("\nXOR Results:")
print(np.round(final_output))
