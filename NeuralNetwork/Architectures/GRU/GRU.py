import numpy as np
from NeuralNetwork.Architectures import Architecture
from NeuralNetwork.Utillities.activation_functions import Sigmoid, Tanh
from NeuralNetwork.Architectures.LSTM.OutputCell import OutputCell
from NeuralNetwork.Architectures.GRU.Cell import Cell
import pandas as pd
from gensim import models, similarities, downloader
import logging
import re
import random
import spacy

ArchitectureType = Architecture.ArchitectureType
Architecture = Architecture.Architecture

INPUT_UNITS = 25

# beta1 for V parameters used in Adam Optimizer
beta1 = 0.90

# beta2 for S parameters used in Adam Optimizer
beta2 = 0.99


class GRU(Architecture):
    # Constructor
    def __init__(self, list_of_feelings, hidden_units=256, learning_rate=0.001, std=0.01, embed=False):
        super().__init__(ArchitectureType.GRU)

        self.loss = [0]
        self.parameters = dict()
        self.input_units = INPUT_UNITS
        self.output_units = len(list_of_feelings)
        self.list_of_feelings = list_of_feelings
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

        self.initialize_parameters(std)

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        if embed:
            self.nlp = spacy.load("en_core_web_sm")
            self.model = downloader.load('glove-twitter-25')

    def initialize_parameters(self, std):
        mean = 0

        input_weights = np.random.normal(mean, std, (self.hidden_units, self.input_units))
        hidden_weights = np.random.normal(mean, std, (self.hidden_units,  self.hidden_units))
        output_weights = np.random.normal(mean, std, (self.output_units,  self.hidden_units))

        # Creating matrix with one colum because we need a vector
        hidden_biases = np.random.normal(mean, std, (self.hidden_units, 1))
        output_biases = np.random.normal(mean, std, (self.hidden_units, 1))

        self.parameters['wx'] = input_weights
        self.parameters['wh'] = hidden_weights
        self.parameters['wo'] = output_weights
        self.parameters['bh'] = hidden_biases
        self.parameters['bo'] = output_biases

        return self.parameters

    def forward_propagation(self, sentence):
        # Store the activations of all the unrollings.
        hidden_cache = dict()


        previous_hidden = np.zeros([self.hidden_units], dtype=np.float32)

        # store the initial activations in cache
        hidden_cache['h0'] = previous_hidden

        for i in range(len(sentence)):

            word = sentence[i]

            # gru cell
            hidden_t = Cell.activate(word, previous_hidden , self.parameters)

            # store the time 't' activations in caches
            hidden_cache['h' + str(i + 1)] = hidden_t

            # update pervious_hidden to the curren hidden
            previous_hidden = hidden_t

        # output cell
        output, softmax = OutputCell.activate(hidden_t, self.parameters)

        return hidden_cache, output, softmax

    # backpropagation
    def backward_propagation(self, sentence, sentence_labels, hidden_cache,output, softmax):
        # calculate output errors
        self.loss, output_weights_error, output_biases_error, hidden_error = OutputCell.calculate_error(sentence_labels,hidden_cache, output , softmax, self.parameters, self.loss)

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
        fgw = fgw - self.learning_rate * ((vfgw) / (np.sqrt(sfgw) + 1e-6))
        igw = igw - self.learning_rate * ((vigw) / (np.sqrt(sigw) + 1e-6))
        ogw = ogw - self.learning_rate * ((vogw) / (np.sqrt(sogw) + 1e-6))
        ggw = ggw - self.learning_rate * ((vggw) / (np.sqrt(sggw) + 1e-6))
        how = how - self.learning_rate * ((vhow) / (np.sqrt(show) + 1e-6))

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

    def run_model_with_embedding(self, input_string):
        regex = re.compile(r'[^a-zA-Z\s]')
        text = regex.sub('', input_string)
        text = text.lower()

        # sentence => array of words
        arr = text.split(" ")

        we_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        words_arr = []

        # words => word embedding
        for word in arr:
            doc = self.nlp(word)
            if doc and doc[0].is_stop:
                continue
            try:
                # print("Before:", word)
                word = doc[0].lemma_
                # print("After:", word)
                word_vec = self.model[word]
                words_arr.append(word_vec)
            except Exception:
                pass
            # print(word + " wasn't found on word embedding.")
        input_data = words_arr

        res = self.run_model(input_data)
        highest = [0, 0]


        for i in range(len(res)):
            if res[i] > highest[0]:
                highest[0] = res[i]
                highest[1] = i
        dict = {}
        for i in range(len(self.list_of_feelings)):
            emotion = self.list_of_feelings[i]
            dict[emotion] = res[i]
        return (self.list_of_feelings[highest[1]],dict)

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
