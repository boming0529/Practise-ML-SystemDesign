# neuron using popo
# POPO - Plain Old Python Object
import numpy as np

def ReLU(z):
    return np.maximum(0, z)

x = np.array([1.0, 2.0, 3.0])  # input layer : 3 

# hidden layer
# first layer （3 input  -> 4 neuron)
W1 = np.array([[0.1, 0.2, 0.3, 0.4],   # weight matrix (3x4)
               [0.5, 0.6, 0.7, 0.8],
               [0.9, 1.0, 1.1, 1.2]])
b1 = np.array([0.1, 0.2, 0.3, 0.4])    # bias (4x1)

# second layer （4 input -> 2 neuron)
W2 = np.array([[0.1, 0.2],              # weight matrix (4x2)
               [0.3, 0.4],
               [0.5, 0.6],
               [0.7, 0.8]])
b2 = np.array([0.1, 0.2])               # bias (2x1)

# output layer （2 input -> 1 neuron）
W3 = np.array([[0.1],                   # weight matrix (2x1)
               [0.2]])
b3 = np.array([0.1])                    # bias (1,)

# forward propagation
# first layer
z1 = np.dot(x, W1) + b1  # Linear eq : x * W1 + b1
a1 = ReLU(z1)            # activation fun：phi(z1)

# second layer
z2 = np.dot(a1, W2) + b2  # Linear eq : a1 * W2 + b2
a2 = ReLU(z2)             # activation fun : phi(z2)

# output layer
z3 = np.dot(a2, W3) + b3  # Linear eq : a2 * W3 + b3
y_hat = z3               

# print
print("input:", x)
print("hidden first layer:", a1)
print("hidden second layer:", a2)
print("output:", y_hat)