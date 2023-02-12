# Abstract Architecture class
from enum import Enum

class ArchitectureType(Enum):
    BASIC = 0
    LSTM = 1
    GRU = 2
    CNN = 3
    NEW_LSTM = 4

class Architecture:
    def __init__(self, arch_type):
        self.type = arch_type

    def train(self, examples, test, batch_size, iters, dataset_name="undefined"):
        raise NotImplementedError

    def run_model(self, input_data):
        raise

    def run_model_with_embedding(self, input_string):
        raise
