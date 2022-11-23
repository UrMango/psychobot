import numpy as np
import Architecture

ArchitectureType = Architecture.ArchitectureType
Architecture = Architecture.Architecture

class LSTM(Architecture):
    # Constructor
    def __init__(self):
        super().__init__(ArchitectureType.LSTM)
        self.layers = []

    def run_model(self, input_data):
        raise NotImplementedError

    def train(self, examples):  # [[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]]]
        raise NotImplementedError
