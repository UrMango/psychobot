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
        super().__init__(ArchitectureType.NEW_LSTM)

        self.loss = []
        self.accuracy = []

        self.std = std
        self.input_units = INPUT_UNITS
        self.output_units = len(list_of_feelings)
        self.list_of_feelings = list_of_feelings
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

        self.output_layers_dict = {}
        self.nudge_layers_dict = {}

        self.layers_dict = {}
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.initialize_layers()
        if embed:
            self.nlp = spacy.load("en_core_web_sm")
            self.model = downloader.load('glove-twitter-25')

    def initialize_layers(self):
        mean = 0

        self.layers_dict["zr"] = MiddleLayer([self.hidden_units, self.input_units], self.hidden_units, self.std,  "zr", ["h-", "x"])
        self.layers_dict["rr"] = MiddleLayer([self.hidden_units, self.input_units], self.hidden_units, self.std,  "rr", ["h-", "x"])

        self.layers_dict["z"] = ActivationLayer(Sigmoid.sigmoid, Sigmoid.derivative_sigmoid_by_func, "z", ["zr"])
        self.layers_dict["r"] = ActivationLayer(Sigmoid.sigmoid, Sigmoid.derivative_sigmoid_by_func, "r", ["rr"])

        self.layers_dict["mrh"] = MultiplyLayer("mrh", ["r", "h-"])

        self.layers_dict["Hr"] = MiddleLayer([self.hidden_units, self.input_units], self.hidden_units, self.std,  "Hr", ["mrh", "x"])

        self.layers_dict["H"] = ActivationLayer(Tanh.tanh, Tanh.tanh_derivative_by_func, "H", ["Hr"])

        self.layers_dict["omz"] = ActivationLayer(OneMinus.one_minus, OneMinus.derivative_one_minus, "omz", ["z"])

        self.layers_dict["momzh"] = MultiplyLayer("momzh", ["omz", "h-"])
        self.layers_dict["mzH"] = MultiplyLayer("mzH", ["z", "H"])

        self.layers_dict["h"] = AddLayer("h", ["momzh", "mzH"])

        self.layers_dict["sr"] = MiddleLayer([self.hidden_units], self.output_units, self.std,  "sr", ["h"])
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

    def print_graph(self):
        avg_loss = list()
        avg_acc = list()
        i = 0
        while i < len(self.loss):
            avg_loss.append(np.mean(self.loss[i:i + 1000]))
            avg_acc.append(np.mean(self.accuracy[i:i + 1000]))
            i += 1000

        plt1 = plt.figure(1)
        plt.plot(list(range(len(avg_loss))), avg_loss)
        plt.xlabel("x")
        plt.ylabel("Loss (Avg of 1000 examples)")
        plt.title("Loss Graph")
        plt.show()

        plt2 = plt.figure(2)
        plt.plot(list(range(len(avg_acc))), avg_acc)
        plt.xlabel("x")
        plt.ylabel("Accuracy (Avg of 1000 examples)")
        plt.title("Accuracy Graph")
        plt.show()

        avg_loss = []
        avg_acc = []
        i = 0
        while i < len(self.loss):
            avg_loss.append(np.mean(self.loss[i:i + 30000]))
            avg_acc.append(np.mean(self.accuracy[i:i + 30000]))
            i += 30000
        plt3 = plt.figure(3)
        plt.plot(list(range(len(avg_loss))), avg_loss)
        plt.xlabel("x")
        plt.ylabel("Loss (Avg of 30000 examples)")
        plt.title("Loss Graph")
        plt.show()

        plt4 = plt.figure(4)
        plt.plot(list(range(len(avg_acc))), avg_acc)
        plt.xlabel("x")
        plt.ylabel("Accuracy (Avg of 30000 examples)")
        plt.title("Accuracy Graph")
        plt.show()

    # train function
    def train(self, train_dataset, epochs):
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

                #avg_loss = 0
                #avg_accuracy = 0
                #for j in range(len(self.loss) - len(batch), len(self.loss)):
                #    avg_loss += self.loss[j]
                #    avg_accuracy += self.accuracy[j]
                #avg_loss = avg_loss / len(batch)
                #avg_accuracy = avg_accuracy / len(batch)
                print('\r' + "Training 💪 - " + "{:.2f}".format(100 * (batch_i+len(train_dataset)*i)/(epochs*len(train_dataset))) + "% | batch: " + str(batch_i+len(train_dataset)*i) + "/" + str(epochs*len(train_dataset)), end="")
                #print("Loss: " + str(avg_loss))
                #print("Accuracy: " + str(avg_accuracy))
        print("If this number don't match you got a problem: " + str(len(self.loss)) + ", " + str(len(self.accuracy)))

        print()
        self.print_graph()

        return self.output_layers_dict
