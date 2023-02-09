import numpy as np
from NeuralNetwork.Architectures import Architecture
from NeuralNetwork.Utillities.MiddleLayer import MiddleLayer
from NeuralNetwork.Utillities.ActivationLayer import ActivationLayer
from NeuralNetwork.Utillities.MultiplyLayer import MultiplyLayer
from NeuralNetwork.Utillities.AddLayer import AddLayer
from NeuralNetwork.Utillities.SoftmaxLayer import SoftmaxLayer
from NeuralNetwork.Utillities.activation_functions import Sigmoid, Tanh, Softmax, OneMinus
from gensim import models, similarities, downloader
import logging
import re
import random
import spacy
import matplotlib.pyplot as plt  #for visualization
import json
import pickle

ArchitectureType = Architecture.ArchitectureType
Architecture = Architecture.Architecture

INPUT_UNITS = 25

# beta1 for V parameters used in Adam Optimizer
beta1 = 0.90

# beta2 for S parameters used in Adam Optimizer
beta2 = 0.99


class GRU(Architecture):
    # Constructor
    def __init__(self, list_of_feelings, hidden_units=256, learning_rate=1, std=0.01, embed=False, set_parameters=False, parameters={}):
        super().__init__(ArchitectureType.NEW_LSTM)

        self.loss = []
        self.accuracy = []

        self.set_parameters = set_parameters
        self.parameters = parameters

        self.std = std
        self.input_units = INPUT_UNITS
        self.output_units = len(list_of_feelings)
        self.list_of_feelings = list_of_feelings
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.accuracy_test = []

        self.output_layers_dict = {}
        self.nudge_layers_dict = {}

        self.amount_true_feel = []
        self.amount_false_feel = []
        self.amount_false_feel_inv = []
        for feel in list_of_feelings:
            self.amount_true_feel.append(0)
            self.amount_false_feel.append(0)
            self.amount_false_feel_inv.append(0)

        self.layers_dict = {}
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.initialize_layers()
        if embed:
            self.nlp = spacy.load("en_core_web_sm")
            self.model = downloader.load('glove-twitter-25')

    def initialize_layers(self):
        mean = 0

        self.layers_dict["zrrr"] = MiddleLayer([self.hidden_units, self.input_units], self.hidden_units, self.std,  "zrrr", ["h-", "x"], self.set_parameters, self.parameters)
        self.layers_dict["rrrr"] = MiddleLayer([self.hidden_units, self.input_units], self.hidden_units, self.std,  "rrrr", ["h-", "x"], self.set_parameters, self.parameters)

        self.layers_dict["zrr"] = ActivationLayer(Sigmoid.sigmoid, Sigmoid.derivative_sigmoid_by_func, "zrr", ["zrrr"])
        self.layers_dict["rrr"] = ActivationLayer(Sigmoid.sigmoid, Sigmoid.derivative_sigmoid_by_func, "rrr", ["rrrr"])

        self.layers_dict["zr"] = MiddleLayer([self.hidden_units, self.input_units], self.hidden_units, self.std, "zr", ["zrr"], self.set_parameters, self.parameters)
        self.layers_dict["rr"] = MiddleLayer([self.hidden_units, self.input_units], self.hidden_units, self.std, "rr", ["rrr"], self.set_parameters, self.parameters)

        self.layers_dict["z"] = ActivationLayer(Sigmoid.sigmoid, Sigmoid.derivative_sigmoid_by_func, "z", ["zr"])
        self.layers_dict["r"] = ActivationLayer(Sigmoid.sigmoid, Sigmoid.derivative_sigmoid_by_func, "r", ["rr"])

        self.layers_dict["mrh"] = MultiplyLayer("mrh", ["r", "h-"])

        self.layers_dict["Hr"] = MiddleLayer([self.hidden_units, self.input_units], self.hidden_units, self.std,  "Hr", ["mrh", "x"], self.set_parameters, self.parameters)

        self.layers_dict["H"] = ActivationLayer(Tanh.tanh, Tanh.tanh_derivative_by_func, "H", ["Hr"])

        self.layers_dict["omz"] = ActivationLayer(OneMinus.one_minus, OneMinus.derivative_one_minus, "omz", ["z"])

        self.layers_dict["momzh"] = MultiplyLayer("momzh", ["omz", "h-"])
        self.layers_dict["mzH"] = MultiplyLayer("mzH", ["z", "H"])

        self.layers_dict["h"] = AddLayer("h", ["momzh", "mzH"])

        self.layers_dict["sr"] = MiddleLayer([self.hidden_units], self.output_units, self.std,  "sr", ["h"], self.set_parameters, self.parameters)
        self.layers_dict["s"] = SoftmaxLayer("s", ["sr"])

    def reset_per_example(self):
        params_keys = ["dzrhw", "drrhw", "dzrxw", "drrxw", "dzrb", "drrb", "dHrmrhw", "dHrxw", "dHrb", "dsrhw", "dsrb"]
        keys = list(self.nudge_layers_dict.keys())
        copy_keys = keys.copy()
        for key in copy_keys:
            if key not in params_keys:
                del self.nudge_layers_dict[key]

    def reset_per_nudge(self):
        self.nudge_layers_dict = {}

    def forward_propagation(self, sentence):
        previous_hidden = np.zeros((1, self.hidden_units), dtype=np.float32)
        previous_C = np.zeros((1, self.hidden_units), dtype=np.float32)

        self.output_layers_dict["h0"] = previous_hidden
        self.output_layers_dict["C0"] = previous_C

        vector_sentence = []
        for t in range(len(sentence)):
            vector_sentence.append(np.zeros((1, len(sentence[0])), dtype=np.float32))  # convert list to matrix with one raw (vector)
            for i in range(len(sentence[t])):
                vector_sentence[t][0][i] = sentence[t][i]

        for t in range(len(sentence)):
            self.output_layers_dict["x"+str(t+1)] = vector_sentence[t]

        layers = ['zr', 'rr', 'z', 'r', 'mrh', 'Hr', 'H', 'omz', 'momzh', 'mzH', 'h']
        for t in range(1, len(sentence)+1):
            for key in layers:
                self.output_layers_dict = self.layers_dict[key].forward_propagation(self.output_layers_dict, t)

        # output cell
        time = len(sentence)  # this is the time of the last cell
        self.output_layers_dict = self.layers_dict["sr"].forward_propagation(self.output_layers_dict, time)
        self.output_layers_dict = self.layers_dict["s"].forward_propagation(self.output_layers_dict, time)

        return self.output_layers_dict["s"][0]

    # backpropagation
    def backward_propagation(self, sentence, sentence_labels):
        loss, accuracy = 0, 0
        self.nudge_layers_dict, loss, accuracy = self.layers_dict["s"].backward_propagation(self.nudge_layers_dict, self.output_layers_dict, sentence_labels, len(sentence))
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.nudge_layers_dict = self.layers_dict["sr"].backward_propagation(self.nudge_layers_dict, self.output_layers_dict, len(sentence))
        layers = ['h', 'mzH', 'momzh', 'omz', 'H', 'Hr', 'mrh', 'r', 'z', 'rr', 'zr']
        for t in range(len(sentence), 0, -1):
            for layer in layers:
                self.nudge_layers_dict = self.layers_dict[layer].backward_propagation(self.nudge_layers_dict, self.output_layers_dict, t)

    def update_parameters(self, size):
        for key in self.layers_dict.keys():
            self.layers_dict[key].nudge(self.nudge_layers_dict, self.learning_rate, size)

    def run_model(self, input_data):
        output = self.forward_propagation(input_data)

        return output

    def test(self, examples):
        test_len = 0
        amount_true = 0
        for batch in examples:
            for example in batch:
                ls = []
                up_index = 0
                for i in range(len(self.list_of_feelings)):
                    ls.append(example[1][i])
                    if i > 0:
                        if ls[i] > ls[i - 1]:
                            up_index = i
                if self.check_input(example[0], self.list_of_feelings[up_index], up_index, self.list_of_feelings):
                    amount_true += 1
                test_len += 1
        return amount_true / test_len
    def check_input(self, input_data, expected_feeling, expected_feeling_index, list_of_feelings):
        return_val = False

        res = self.run_model(input_data)
        highest = [0, 0]

        for i in range(len(res)):
            if res[i] > highest[0]:
                highest[0] = res[i]
                highest[1] = i

        if str(list_of_feelings[highest[1]]) == str(expected_feeling):
            self.amount_true_feel[highest[1]] += 1
            return_val = True
        else:
            self.amount_false_feel_inv[highest[1]] += 1
            self.amount_false_feel[expected_feeling_index] += 1

        return return_val

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
        return self.list_of_feelings[highest[1]], dict

    def print_graph(self, epochs, examples, accuracy_test):
        avg_loss = list()
        avg_acc = list()
        i = 0
        while i < len(self.loss):
            avg_loss.append(np.mean(self.loss[i:i + 10*examples]))
            avg_acc.append(np.mean(self.accuracy[i:i + 10*examples]))
            i += 10*examples

        plt1 = plt.figure(1)
        plt.plot(list(range(len(avg_loss))), avg_loss)
        plt.xlabel("x")
        plt.ylabel("Loss (Avg of 10 batches)")
        plt.title("Loss Graph Per Batch, learning rate: "+str(self.learning_rate)+" batch_size: "+str(examples))
        plt.show()

        plt2 = plt.figure(2)
        plt.plot(list(range(len(avg_acc))), avg_acc)
        plt.xlabel("x")
        plt.ylabel("Accuracy (Avg of 10 batches)")
        plt.title("Accuracy Graph Per Batch, learning rate: "+str(self.learning_rate)+" batch_size: "+str(examples))
        plt.show()

        avg_loss = []
        avg_acc = []
        i = 0
        examples_in_epoch = len(self.loss) / epochs
        examples_in_epoch = int(examples_in_epoch)
        while i < len(self.loss):
            avg_loss.append(np.mean(self.loss[i:i + examples_in_epoch]))
            avg_acc.append(np.mean(self.accuracy[i:i + examples_in_epoch]))
            i += examples_in_epoch
        plt3 = plt.figure(3)
        plt.plot(list(range(len(avg_loss))), avg_loss)
        plt.xlabel("x")
        plt.ylabel("Loss (Avg of epoch)")
        plt.title("Loss Graph Per Epoch, learning rate: "+str(self.learning_rate)+" batch_size: "+str(examples))
        plt.show()

        plt4 = plt.figure(4)
        plt.plot(list(range(len(avg_acc))), avg_acc)
        plt.xlabel("x")
        plt.ylabel("Accuracy (Avg of epoch)")
        plt.title("Accuracy Graph Per Epoch, learning rate: "+str(self.learning_rate)+" batch_size: "+str(examples))
        plt.show()

        avg_acc = accuracy_test
        plt5 = plt.figure(5)
        plt.plot(list(range(len(avg_acc))), avg_acc)
        plt.xlabel("x")
        plt.ylabel("Accuracy (Avg of epoch)")
        plt.title("Accuracy On Test Data, learning rate: " + str(self.learning_rate) + " batch_size: " + str(examples))
        plt.show()

    def save_parameters(self):
        dict_parameters = {}
        for key in self.layers_dict.keys():
            dict_parameters = self.layers_dict[key].save_parameters(dict_parameters)
        with open("parameters_"+str(self.list_of_feelings)+".json", 'wb') as f:
            pickle.dump(dict_parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

    # train function
    def train(self, train_dataset, test_dataset, epochs):
        valid = 0
        for i in range(epochs):
            batch_i = -1
            for batch in train_dataset:
                batch_i += 1
                example_k = -1
                for example in batch:
                    example_k += 1
                    if len(example[0]) == 0:
                        continue

                    # forward propagation

                    softmax = self.forward_propagation(example[0])

                    self.backward_propagation(example[0], example[1])
                    self.reset_per_example()
                self.update_parameters(len(batch))
                self.reset_per_nudge()

                print('\r' + "Training 💪 - " + "{:.2f}".format(100 * (1+batch_i+len(train_dataset)*i)/(epochs*len(train_dataset))) + "% | batch: " + str(1+batch_i+len(train_dataset)*i) + "/" + str(epochs*len(train_dataset)), end="")
            self.accuracy_test.append(self.test(test_dataset))
            if self.accuracy_test[-1] > valid:
                valid = self.accuracy_test[-1]
                self.save_parameters()

        print()
        self.print_graph(epochs, len(train_dataset[0]), self.accuracy_test)
        self.save_parameters()
        return self.accuracy_test
