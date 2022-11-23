import numpy as np
from NeuralNetwork.Utillities import Cost


class NeuralNetwork:
    # Constructor
    def __init__(self, architecture):
        self.architecture = architecture
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def run_model(self, input_data):
        return self.architecture.run_model()

    def train(self, examples):
        self.architecture.train()

    def average_cost(self, examples):
        costs = []
        for example in examples:
            current_output = self.run_model(np.array(example[0]))
            costs.append(Cost.cost(current_output, np.array(example[1])))
        cost_average = np.mean(costs)

        return cost_average
