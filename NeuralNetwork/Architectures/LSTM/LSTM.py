import numpy as np
from NeuralNetwork.Architectures import Architecture
from NeuralNetwork.Utillities.activation_functions import Sigmoid, Tanh
from NeuralNetwork.Architectures.LSTM.OutputCell import OutputCell
from NeuralNetwork.Architectures.LSTM.Cell import Cell

ArchitectureType = Architecture.ArchitectureType
Architecture = Architecture.Architecture

# learning rate
learning_rate = 0.001

# beta1 for V parameters used in Adam Optimizer
beta1 = 0.90

# beta2 for S parameters used in Adam Optimizer
beta2 = 0.99
HIDDEN_UNITS = 256


class LSTM(Architecture):
    # Constructor
    def __init__(self,list_of_feelings):
        super().__init__(ArchitectureType.LSTM)

        self.parameters = dict()

        self.input_units = 25
        self.output_units = len(list_of_feelings)
        self.hidden_units = HIDDEN_UNITS

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

    def forward_propagation(self, sentence):

        word_size = len(sentence[0])

        # to store the activations of all the unrollings.
        lstm_cache = dict()
        activation_cache = dict()
        cell_cache = dict()
        output_cache = dict()
        embedding_cache = dict()

        # initial activation_matrix(a0) and cell_matrix(c0)
        a0 = np.zeros([HIDDEN_UNITS], dtype=np.float32)
        c0 = np.zeros([HIDDEN_UNITS], dtype=np.float32)

        # store the initial activations in cache
        activation_cache['a0'] = a0
        cell_cache['c0'] = c0

        output_at = None

        # unroll the names
        for i in range(len(sentence) - 1):
            # get first character batch
            word = sentence[i]

            # lstm cell
            lstm_activations, ct, at = Cell.activate(word, a0, c0, self.parameters)

            output_at = at
            # print(output_at)

            # store the time 't' activations in caches
            lstm_cache['lstm' + str(i + 1)] = lstm_activations
            activation_cache['a' + str(i + 1)] = at
            cell_cache['c' + str(i + 1)] = ct

            # update a0 and c0 to new 'at' and 'ct' for next lstm cell
            a0 = at
            c0 = ct

        # output cell
        ot = OutputCell.activate(output_at, self.parameters)

        output_cache['o'] = ot

        return lstm_cache, activation_cache, cell_cache, output_cache

    # backpropagation
    def backward_propagation(self, sentence, sentence_labels, lstm_cache, activation_cache, cell_cache, output_cache):
        # calculate output errors
        output_error_cache, activation_error_cache = OutputCell.calculate_error(sentence_labels, output_cache,
                                                                                self.parameters)

        # to store lstm error for each time step
        lstm_error_cache = dict()

        # next activation error
        # next cell error
        # for last cell will be zero
        eat = np.zeros(activation_error_cache['ea'].shape)
        ect = np.zeros(activation_error_cache['ea'].shape)

        # calculate all lstm cell errors (going from last time-step to the first time step)
        for i in range(len(lstm_cache), 0, -1):
            # calculate the lstm errors for this time step 't'
            pae, pce, le = Cell.calculate_error(activation_error_cache['ea'], eat, ect,
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
                activation_cache['a' + str(i - 1)],
                len(sentence)
            )

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

    def update_parameters(self, derivatives, V, S, t):
        # get derivatives
        dfgw = derivatives['dfgw']
        digw = derivatives['digw']
        dogw = derivatives['dogw']
        dggw = derivatives['dggw']
        dhow = derivatives['dhow']

        # get parameters
        fgw = self.parameters['fgw']
        igw = self.parameters['igw']
        ogw = self.parameters['ogw']
        ggw = self.parameters['ggw']
        how = self.parameters['how']

        # get V parameters
        vfgw = V['vfgw']
        vigw = V['vigw']
        vogw = V['vogw']
        vggw = V['vggw']
        vhow = V['vhow']

        # get S parameters
        sfgw = S['sfgw']
        sigw = S['sigw']
        sogw = S['sogw']
        sggw = S['sggw']
        show = S['show']

        # calculate the V parameters from V and current derivatives
        vfgw = (beta1 * vfgw + (1 - beta1) * dfgw)
        vigw = (beta1 * vigw + (1 - beta1) * digw)
        vogw = (beta1 * vogw + (1 - beta1) * dogw)
        vggw = (beta1 * vggw + (1 - beta1) * dggw)
        vhow = (beta1 * vhow + (1 - beta1) * dhow)

        # calculate the S parameters from S and current derivatives
        sfgw = (beta2 * sfgw + (1 - beta2) * (dfgw ** 2))
        sigw = (beta2 * sigw + (1 - beta2) * (digw ** 2))
        sogw = (beta2 * sogw + (1 - beta2) * (dogw ** 2))
        sggw = (beta2 * sggw + (1 - beta2) * (dggw ** 2))
        show = (beta2 * show + (1 - beta2) * (dhow ** 2))

        # update the parameters
        fgw = fgw - learning_rate * ((vfgw) / (np.sqrt(sfgw) + 1e-6))
        igw = igw - learning_rate * ((vigw) / (np.sqrt(sigw) + 1e-6))
        ogw = ogw - learning_rate * ((vogw) / (np.sqrt(sogw) + 1e-6))
        ggw = ggw - learning_rate * ((vggw) / (np.sqrt(sggw) + 1e-6))
        how = how - learning_rate * ((vhow) / (np.sqrt(show) + 1e-6))

        # store the new weights
        self.parameters['fgw'] = fgw
        self.parameters['igw'] = igw
        self.parameters['ogw'] = ogw
        self.parameters['ggw'] = ggw
        self.parameters['how'] = how

        # store the new V parameters
        V['vfgw'] = vfgw
        V['vigw'] = vigw
        V['vogw'] = vogw
        V['vggw'] = vggw
        V['vhow'] = vhow

        # store the s parameters
        S['sfgw'] = sfgw
        S['sigw'] = sigw
        S['sogw'] = sogw
        S['sggw'] = sggw
        S['show'] = show

        return V, S

    def initialize_V(self):
        Vfgw = np.zeros(self.parameters['fgw'].shape)
        Vigw = np.zeros(self.parameters['igw'].shape)
        Vogw = np.zeros(self.parameters['ogw'].shape)
        Vggw = np.zeros(self.parameters['ggw'].shape)
        Vhow = np.zeros(self.parameters['how'].shape)

        V = dict()
        V['vfgw'] = Vfgw
        V['vigw'] = Vigw
        V['vogw'] = Vogw
        V['vggw'] = Vggw
        V['vhow'] = Vhow
        return V

    def initialize_S(self):
        Sfgw = np.zeros(self.parameters['fgw'].shape)
        Sigw = np.zeros(self.parameters['igw'].shape)
        Sogw = np.zeros(self.parameters['ogw'].shape)
        Sggw = np.zeros(self.parameters['ggw'].shape)
        Show = np.zeros(self.parameters['how'].shape)

        S = dict()
        S['sfgw'] = Sfgw
        S['sigw'] = Sigw
        S['sogw'] = Sogw
        S['sggw'] = Sggw
        S['show'] = Show
        return S

    def run_model(self, input_data):
        lstm_cache, activation_cache, cell_cache, output_cache = self.forward_propagation(input_data)

        return output_cache['o']

    # train function
    def train(self, train_dataset, iters=1000):

        # initialize the V and S parameters for Adam
        V = self.initialize_V()
        S = self.initialize_S()

        for step in range(iters):
            # get batch dataset
            index = step % len(train_dataset)
            sentence = train_dataset[index]

            if len(sentence) == 0:
                continue

            # forward propagation
            lstm_cache, activation_cache, cell_cache, output_cache = self.forward_propagation(sentence[0])

            # backward propagation
            derivatives = self.backward_propagation(sentence[0], sentence[1], lstm_cache, activation_cache, cell_cache, output_cache)

            # update the parameters
            V, S = self.update_parameters(derivatives, V, S, step)
            print('\r' + "Training LSTM 💪 - " + "{:.2f}".format(100 * (step / iters)) + "% | example: " + str(
                step) + "/" + str(iters), end="")

        return self.parameters
