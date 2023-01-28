# Abstract Layer class
from enum import Enum

class LayerType(Enum):
    NONE = 0
    ACTIVATION = 1
    MIDDLE = 2
    MULTIPLY = 3
    SOFTMAX = 4
    ADD = 5

class Layer:
    def __init__(self):
        self.inputs_id = None  # All the id of layers that this layer get as input
        self.output = None

        self.type = LayerType.NONE
        self.id = None

    # def forward_propagation(self, output_layers_dict, time):
    #     raise NotImplementedError

    def nudge(self, nudge_layers_dict, learning_rate, batch_len):
        raise NotImplementedError

    def save_parameters(self, parameters_dict):
        raise NotImplementedError
