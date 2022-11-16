# Abstract Layer class
from enum import Enum

class LayerType(Enum):
    NONE = 0
    ACTIVATION = 1
    MIDDLE = 2

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.type = LayerType.NONE

    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backward_propagation(self, output_nudge):
        raise NotImplementedError
