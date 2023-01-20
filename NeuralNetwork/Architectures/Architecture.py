# Abstract Architecture class
from enum import Enum

class ArchitectureType(Enum):
    BASIC = 0
    LSTM = 1
    GRU = 2
    CNN = 3

class Architecture:
    def __init__(self, arch_type):
        self.type = arch_type

    def train(self, examples, iters):
        raise NotImplementedError

    def run_model(self, input_data):
        raise

    def run_model_with_embedding(self, input_string):
        raise
