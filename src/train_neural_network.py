import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Get New Saving Directory (dataset will be saved there)

# Load Dataset
df = pd.read_csv('../datasets/depol_error/U3_13_13.csv')
X_df = df[['theta', 'phi', 'lam', 'E']]
y_df = df['p']

X = torch.tensor(np.array(X_df), dtype=torch.float32)
y = torch.tensor(y_df, dtype=torch.float32).unsqueeze(1)
xPredicted = torch.tensor([0.0, 0.0, 0.0, 0.1], dtype=torch.float32)

# scaling
X_max, _ = torch.max(X, 0)
xPredicted_max, _ = torch.max(xPredicted, 0)
y_max, _ = torch.max(y, 0)
X = torch.div(X, X_max)
xPredicted = torch.div(xPredicted, xPredicted_max)
y = torch.div(y, y_max)


class Neural_Network(nn.Module):
    def __init__(self, inputSize=4):
        super(Neural_Network, self).__init__()
        # parameters
        self.inputSize = inputSize
        self.outputSize = 1
        self.hiddenSize = 3
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)

        # todo: add bias

    def forward(self, X):
        self.z = torch.matmul(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self, model):
        torch.save(model, "NN")
#         torch.load("NN")

    def predict(self):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(xPredicted))
        print("Output: \n" + str(self.forward(xPredicted)))

NN = Neural_Network()
for i in range(1000):  # trains the NN 1,000 times
    print("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X)) ** 2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
NN.saveWeights(NN)
NN.predict()
