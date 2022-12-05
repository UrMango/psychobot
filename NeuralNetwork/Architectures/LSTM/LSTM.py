import numpy as np
from NeuralNetwork.Architectures import Architecture
from NeuralNetwork.Utillities.activation_functions import Sigmoid, Tanh
from NeuralNetwork.Architectures.LSTM.OutputCell import OutputCell

ArchitectureType = Architecture.ArchitectureType
Architecture = Architecture.Architecture

HIDDEN_UNITS = 0

class LSTM(Architecture):
    # Constructor
    def __init__(self):
        super().__init__(ArchitectureType.LSTM)

        self.parameters = dict()

        self.input_units = 10
        self.output_units = 20
        self.hidden_units = 10

        self.initialize_parameters()

    def initialize_parameters(self):
        mean = 0
        std = 0.01
        
        # gates
        forget_gate_weights = np.random.normal(mean, std, (self.input_units + self.hidden_units, self.hidden_units))
        input_gate_weights = np.random.normal(mean, std, (self.input_units + self.hidden_units, self.hidden_units))
        output_gate_weights = np.random.normal(mean, std, (self.input_units + self.hidden_units, self.hidden_units))
        gate_gate_weights = np.random.normal(mean, std, (self.input_units + self.hidden_units, self.hidden_units))

        # hidden to output weights
        hidden_output_weights = np.random.normal(mean, std, (self.hidden_units, self.output_units))

        self.parameters['fgw'] = forget_gate_weights
        self.parameters['igw'] = input_gate_weights
        self.parameters['ogw'] = output_gate_weights
        self.parameters['ggw'] = gate_gate_weights
        self.parameters['how'] = hidden_output_weights

        return self.parameters

    def forward_propagation(self, sentence, parameters):
        # get batch size
        batch_size = sentence[0].shape[0]

        # to store the activations of all the unrollings.
        lstm_cache = dict()
        activation_cache = dict()
        cell_cache = dict()
        output_cache = dict()
        embedding_cache = dict()

        # initial activation_matrix(a0) and cell_matrix(c0)
        a0 = np.zeros([batch_size, HIDDEN_UNITS], dtype=np.float32)
        c0 = np.zeros([batch_size, HIDDEN_UNITS], dtype=np.float32)

        # store the initial activations in cache
        activation_cache['a0'] = a0
        cell_cache['c0'] = c0

        output_at = None

        # unroll the names
        for i in range(len(sentence) - 1):
            # get first character batch
            word = sentence[i]

            # lstm cell
            lstm_activations, ct, at = Cell.activate_cell(word, a0, c0, parameters)

            output_at = at

            # store the time 't' activations in caches
            lstm_cache['lstm' + str(i + 1)] = lstm_activations
            activation_cache['a' + str(i + 1)] = at
            cell_cache['c' + str(i + 1)] = ct

            # update a0 and c0 to new 'at' and 'ct' for next lstm cell
            a0 = at
            c0 = ct

        # output cell
        ot = OutputCell.activate(output_at, parameters)

        output_cache['o'] = ot

        return embedding_cache, lstm_cache, activation_cache, cell_cache, output_cache

    # backpropagation
    def backward_propagation(self, sentence_labels, lstm_cache, activation_cache, cell_cache, output_cache):
        # calculate output errors
        output_error_cache, activation_error_cache = OutputCell.calculate_error(sentence_labels, output_cache, self.parameters)

        # to store lstm error for each time step
        lstm_error_cache = dict()

        # next activation error
        # next cell error
        # for last cell will be zero
        eat = np.zeros(activation_error_cache['ea1'].shape)
        ect = np.zeros(activation_error_cache['ea1'].shape)

        # calculate all lstm cell errors (going from last time-step to the first time step)
        for i in range(len(lstm_cache), 0, -1):
            # calculate the lstm errors for this time step 't'
            pae, pce, ee, le = Cell.calculate_error(activation_error_cache['ea' + str(i)], eat, ect,
                                                                self.parameters, lstm_cache['lstm' + str(i)],
                                                                cell_cache['c' + str(i)], cell_cache['c' + str(i - 1)])

            # store the lstm error in dict
            lstm_error_cache['elstm' + str(i)] = le

            # update the next activation error and next cell error for previous cell
            eat = pae
            ect = pce

        # calculate output cell derivatives
        derivatives = dict()
        derivatives['dhow'] = OutputCell.calculate_derivatives(output_error_cache, activation_cache, self.parameters)

        # calculate lstm cell derivatives for each time step and store in lstm_derivatives dict
        lstm_derivatives = dict()
        for i in range(1, len(lstm_error_cache) + 1):
            lstm_derivatives['dlstm' + str(i)] = Cell.calculate_derivatives(
                lstm_error_cache['elstm' + str(i)],
                activation_cache['a' + str(i - 1)])

        # initialize the derivatives to zeros
        derivatives['dfgw'] = np.zeros(self.parameters['fgw'].shape)
        derivatives['digw'] = np.zeros(self.parameters['igw'].shape)
        derivatives['dogw'] = np.zeros(self.parameters['ogw'].shape)
        derivatives['dggw'] = np.zeros(self.parameters['ggw'].shape)

        # sum up the derivatives for each time step
        for i in range(1, len(lstm_error_cache) + 1):
            derivatives['dfgw'] += lstm_derivatives['dlstm' + str(i)]['dfgw']
            derivatives['digw'] += lstm_derivatives['dlstm' + str(i)]['digw']
            derivatives['dogw'] += lstm_derivatives['dlstm' + str(i)]['dogw']
            derivatives['dggw'] += lstm_derivatives['dlstm' + str(i)]['dggw']

        return derivatives
    
    def run_model(self, input_data, layers):
        raise NotImplementedError

    def train(self, examples, layers):  # [[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]]]
        raise NotImplementedError
