# Neural Network: Feed Forward Network Implementation

This project demonstrates a simple feed-forward neural network with one hidden layer, using both linear transformations and activation functions (Tanh and ReLU). The model takes input data, applies weighted sums, and generates outputs based on the hidden layer's computations.


## Introduction

The goal of this project is to implement a basic feed-forward neural network from scratch in Python. The network consists of:
- An input layer with 2 nodes
- A hidden layer with 2 nodes
- An output layer with 1 node

The network computes output values both with and without activation functions such as Tanh and ReLU, helping to capture non-linear relationships in the data.


## Code Explanation

### Input Data and Weights

We start by defining the input data and initializing the weights for the hidden layer and output layer:

```python
input_data = np.array([2, 3])

weights = {
    'node0': np.array([1, 1]),
    'node1': np.array([-1, 1]),
    'output': np.array([2, -1])
}
```

### Hidden Layer Computation

Each node in the hidden layer computes a weighted sum of the input values:

```python
node0_value = (input_data * weights['node0']).sum()
node1_value = (input_data * weights['node1']).sum()
```

### Output Layer Computation

The output of the hidden layer is used to compute the final output value by performing a weighted sum:

```python
output = (hidden_layer_value * weights['output']).sum()
```

### Applying Activation Functions

Activation functions help capture non-linearities in the model. We demonstrate using the Tanh and ReLU functions.

For Tanh activation:
```python
node0_output = np.tanh(node0_input)
node1_output = np.tanh(node1_input)
```

For ReLU activation:
```python
def relu(input):
    return max(input, 0)
```

## Predict Function

The `predict_with_network()` function takes in input data, computes hidden layer outputs using the ReLU activation function, and calculates the final output for the model:

```python
def predict_with_network(input_data_row, weights):
    node_0_output = relu((input_data_row * weights['node0']).sum())
    node_1_output = relu((input_data_row * weights['node1']).sum())
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    return relu((hidden_layer_outputs * weights['output']).sum())
```

## Results

The results are computed for each input in `input_data`. Here’s an example of the final output:

```bash
[8, 12]
```

These results show the predicted outputs of the model based on the input data after passing through the network.
