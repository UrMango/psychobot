# Abstract Architecture class
from enum import Enum

class ArchitectureType(Enum):
    BASIC = 0
    LSTM = 1
    RNN = 2
    CNN = 3

class Architecture:
    def __init__(self, arch_type):
        self.type = arch_type

    def train(self, examples, layers):
        raise NotImplementedError

    def run_model(self, input_data, layers):
        raise NotImplementedError
