import numpy as np
from NeuralNetwork.Architectures import Architecture
from NeuralNetwork.Utillities.activation_functions import Sigmoid, Tanh

ArchitectureType = Architecture.ArchitectureType
Architecture = Architecture.Architecture


class LSTM(Architecture):
    # Constructor
    def __init__(self):
        super().__init__(ArchitectureType.LSTM)
        self.initialize_parameters(10, 20, 10)
    
    def initialize_parameters(self, input_units, output_units, hidden_units):
        mean = 0
        std = 0.01
        
        # gates
        forget_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
        input_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
        output_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
        gate_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))

        # hidden to output weights
        hidden_output_weights = np.random.normal(mean, std, (hidden_units, output_units))

        parameters = dict()
        parameters['fgw'] = forget_gate_weights
        parameters['igw'] = input_gate_weights
        parameters['ogw'] = output_gate_weights
        parameters['ggw'] = gate_gate_weights
        parameters['how'] = hidden_output_weights

        return parameters
    
    def cell_propagation(self, batch, prev_activation_mat, prev_cell_mat, params):
        # get parameters
        fgw = params['fgw']
        igw = params['igw']
        ogw = params['ogw']
        ggw = params['ggw']

        # concat batch data and prev_activation matrix
        concat_dataset = np.concatenate((batch, prev_activation_mat), axis=1)

        # forget gate activations
        fa = np.matmul(concat_dataset, fgw)
        fa = Sigmoid.sigmoid(fa)

        # input gate activations
        ia = np.matmul(concat_dataset, igw)
        ia = Sigmoid.sigmoid(ia)

        # output gate activations
        oa = np.matmul(concat_dataset, ogw)
        oa = Sigmoid.sigmoid(oa)

        # gate gate activations
        ga = np.matmul(concat_dataset, ggw)
        ga = Tanh.tanh(ga)

        # new cell memory matrix
        cell_memory_matrix = np.multiply(fa, prev_cell_mat) + np.multiply(ia, ga)

        # current activation matrix
        activation_matrix = np.multiply(oa, Tanh.tanh(cell_memory_matrix))

        # lets store the activations to be used in back prop
        lstm_activations = dict()
        lstm_activations['fa'] = fa
        lstm_activations['ia'] = ia
        lstm_activations['oa'] = oa
        lstm_activations['ga'] = ga
    
        return lstm_activations, cell_memory_matrix, activation_matrix
    
    
    def run_model(self, input_data, layers):
        raise NotImplementedError

    def train(self, examples, layers):  # [[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]]]
        raise NotImplementedError
