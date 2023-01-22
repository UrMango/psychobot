# Abstract Layer class
from enum import Enum

class LayerType(Enum):
    NONE = 0
    ACTIVATION = 1
    MIDDLE = 2
    MULTIPLY = 3
    SOFTMAX = 4

class Layer:
    def __init__(self):
        self.inputs_id = None #All the id of layers that this layer get as input
        self.outputs_id = None #All the id of the layers that get this layer as input
        self.output = None

        self.type = LayerType.NONE
        self.id = None

    def forward_propagation(self, output_layers_dict, time):
        raise NotImplementedError

    def backward_propagation(self, output_nudge):
        raise NotImplementedError

    def update_parameters(self, dict_nudge):
        raise NotImplementedError
