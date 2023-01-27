import numpy as np
from NeuralNetwork.Utillities import Cost

class NeuralNetwork:
    # Constructor
    def __init__(self, architecture):
        self.architecture = architecture

    def add_layer(self, layer):
        if self.architecture.type == ArchitectureType.BASIC:
            self.architecture.addLayer(layer)

    def run_model(self, input_data):
        return self.architecture.run_model(input_data)

    def run_model_with_embedding(self, input_string):
        return self.architecture.run_model_with_embedding(input_string)

    def train(self, training_examples, test_examples, iters):
        self.architecture.train(training_examples, test_examples, iters)

    def average_cost(self, examples):
        costs = []
        for example in examples:
            current_output = self.run_model(np.array(example[0]))
            costs.append(Cost.cost(current_output, np.array(example[1])))
        cost_average = np.mean(costs)

        return cost_average
