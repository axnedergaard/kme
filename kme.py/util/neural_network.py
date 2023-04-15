import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, activation_functions, sampler=None):
        super(NeuralNetwork, self).__init__()

        self.sampler = sampler
        self.layers = nn.ModuleList()
        self.activations = activation_functions

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activations[i](x)
        return x

    def reset_weights(self, variance):
        if self.sampler:
            for layer in self.layers:
                self.sampler.sample_normal_matrix(layer.weight, variance)
                self.sampler.sample_normal_matrix(layer.bias, variance)

    def train_network(self, inputs, targets, learning_rate, epochs):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
