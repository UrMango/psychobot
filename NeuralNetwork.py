import numpy as np
import Cost

class NeuralNetwork:
    # Constructor
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def run_model(self, input_data):
        input_data_ = input_data.copy()
        for layer in self.layers:
            input_data_ = layer.forward_propagation(input_data_)
        return input_data_

    def train(self, examples):  # [[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]]]
        for example in examples:
            current_output = self.run_model(np.array(example[0]))
            nudge = Cost.derivative_cost(current_output, np.array(example[1]))
            for layer in reversed(self.layers):
                nudge = layer.backward_propagation(nudge)
        print("Cost after one train: " + str(self.average_cost(examples)))

    def average_cost(self, examples):
        costs = []
        for example in examples:
            current_output = self.run_model(np.array(example[0]))
            costs.append(Cost.cost(current_output, np.array(example[1])))
        cost_average = np.mean(costs)

        return cost_average
