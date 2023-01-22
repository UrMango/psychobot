import numpy as np
from NeuralNetwork.Architectures import Architecture
from NeuralNetwork.Utillities.activation_functions import Sigmoid, Tanh
from NeuralNetwork.Architectures.GRU.OutputCell import OutputCell
from NeuralNetwork.Architectures.GRU.Cell import Cell
import pandas as pd
from gensim import models, similarities, downloader
import logging
import re
import random
import spacy
import matplotlib.pyplot as plt  #for visualization


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

        self.loss = []
        self.accuracy = []
        self.parameters = dict()
        self.input_units = INPUT_UNITS
        self.output_units = len(list_of_feelings)
        self.list_of_feelings = list_of_feelings
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.error_dict = {}

        self.initialize_parameters(std)

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        if embed:
            self.nlp = spacy.load("en_core_web_sm")
            self.model = downloader.load('glove-twitter-25')

    def initialize_parameters(self, std):
        mean = 0

        input_weights = np.random.normal(mean, std, (self.hidden_units, self.input_units))
        hidden_weights = np.random.normal(mean, std, (self.hidden_units, self.hidden_units))
        output_weights = np.random.normal(mean, std, (self.output_units, self.hidden_units))

        # Creating matrix with one colum because we need a vector
        hidden_biases = np.random.normal(mean, std, self.hidden_units)
        output_biases = np.random.normal(mean, std, self.output_units)

        self.parameters['iw'] = input_weights
        self.parameters['hw'] = hidden_weights
        self.parameters['ow'] = output_weights
        self.parameters['hb'] = hidden_biases
        self.parameters['ob'] = output_biases
        self.error_dict['iwe'] = 0
        self.error_dict['hwe'] = 0
        self.error_dict['owe'] = 0
        self.error_dict['hbe'] = 0
        self.error_dict['obe'] = 0
        return self.parameters

    def forward_propagation(self, sentence):
        previous_hidden = np.zeros([self.hidden_units], dtype=np.float32)

        # Store the activations of all the unrollings.
        # store the initial activations in cache
        hidden_cache = [previous_hidden]

        for i in range(len(sentence)):
            word = sentence[i]

            # gru cell
            hidden_t = Cell.activate(word, previous_hidden, self.parameters)

            # store the time 't' activations in caches
            hidden_cache.append(hidden_t)

            # update pervious_hidden to the curren hidden
            previous_hidden = hidden_t

        # output cell
        output, softmax = OutputCell.activate(hidden_cache[-1], self.parameters)

        return hidden_cache, output, softmax

    # backpropagation
    def backward_propagation(self, sentence, sentence_labels, hidden_cache, output, softmax):
        # calculate output errors
        output_weights_error, output_biases_error, hidden_error, self.loss, self.accuracy = OutputCell.calculate_error(
            sentence_labels, hidden_cache, output, softmax, self.parameters, self.loss, self.accuracy)

        self.error_dict["owe"] += output_weights_error
        self.error_dict["obe"] += output_biases_error

        for i in range(len(sentence), 0, -1):
            before_hidden = hidden_cache[i-1]
            hidden_error, input_weights_error, hidden_weights_error, hidden_biases_error = Cell.calculate_error(
                                            sentence[i-1], hidden_error, hidden_cache[i], before_hidden, self.parameters)

            self.error_dict["iwe"] += input_weights_error
            self.error_dict["hwe"] += hidden_weights_error
            self.error_dict["hbe"] += hidden_biases_error

    def update_parameters(self, number_of_examples):
        for key in self.error_dict.keys():
            self.error_dict[key] = self.error_dict[key] / number_of_examples

        for key in self.parameters.keys():
            self.parameters[key] -= self.learning_rate * self.error_dict[key+"e"]
            self.error_dict[key + "e"] = 0

    def run_model(self, input_data):
        hidden_cache, output, softmax = self.forward_propagation(input_data)

        return output

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
        return (self.list_of_feelings[highest[1]], dict)

    def print_graph(self):
        avg_loss = list()
        avg_acc = list()
        i = 0
        while i < len(self.loss):
            avg_loss.append(np.mean(self.loss[i:i + 300]))
            avg_acc.append(np.mean(self.accuracy[i:i + 300]))
            i += 300

        plt.plot(list(range(len(avg_loss))), avg_loss)
        plt.xlabel("x")
        plt.ylabel("Loss (Avg of 30 batches)")
        plt.title("Loss Graph")
        plt.show()

        plt.plot(list(range(len(avg_acc))), avg_acc)
        plt.xlabel("x")
        plt.ylabel("Accuracy (Avg of 30 batches)")
        plt.title("Accuracy Graph")
        plt.show()

    # train function
    def train(self, train_dataset, epochs):
        for i in range(epochs):
            batch_i = -1
            for batch in train_dataset:
                batch_i += 1
                print()
                print("Number of Epochs: " + str(i))
                print("Number of Batch: " + str(batch_i))
                example_k = -1
                for example in batch:
                    example_k += 1
                    if len(example[0]) == 0:
                        continue

                    # forward propagation
                    hidden_cache, output, softmax = self.forward_propagation(example[0])

                    # backward propagation #sentence, sentence_labels, hidden_cache, output, softmax
                    self.backward_propagation(example[0], example[1], hidden_cache,output, softmax)

                    # print('\r' + "Training LSTM 💪 - " +
                    #       "Batches: {:.2f}".format(100 * (batch_i / len(train_dataset))) +
                    #       "% / batch: " + str(batch_i) + "/" + str(len(train_dataset)))
                          # " | Examples: {:.2f}".format(100 * (example_k / len(batch))) +
                          # "% / example: " + str(example_k) + "/" + str(len(batch)), end="")
                self.update_parameters(len(batch))
                avg_loss = 0
                avg_accuracy = 0
                for j in range(len(self.loss) - len(batch), len(self.loss)):
                    avg_loss += self.loss[j]
                    avg_accuracy += self.accuracy[j]
                avg_loss = avg_loss / len(batch)
                avg_accuracy = avg_accuracy / len(batch)

                print("For this batch")
                print("Loss: " + str(avg_loss))
                print("Accuracy: " + str(avg_accuracy))
        print("If this number don't match you got a problem: " + str(len(self.loss)) + ", " + str(len(self.accuracy)))

        self.print_graph()

        return self.parameters
