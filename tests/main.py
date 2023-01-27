# Help:
# matrix multiplication - np.dot
# initialize array - np.zeros
# initialize random values array - np.random.uniform
import json
import time
import math
import numpy as np
import spacy

from Dataset.dataset_loader import Dataset

from NeuralNetwork.Architectures.Basic import Basic
from NeuralNetwork.Architectures.LSTM.LSTM import LSTM
from NeuralNetwork.Architectures.NEW_LSTM.NEW_LSTM import NEW_LSTM
from NeuralNetwork.Architectures.GRU.GRU import GRU

import random
from NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuralNetwork.Architectures.Architecture import ArchitectureType
from gensim import downloader

import re

EPOCHS = 10

NUMBER_OF_EXAMPLES_IN_BATCH = 100

EXAMPLES = 30000

TRAINING_SET_PERCENTAGE = 0.8

nlp = spacy.load("en_core_web_sm")

def accuracy(right, current):
    return 100 - np.abs(((right - current) / right) * 100)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def machine_with_params(list_of_feelings):
    learning_rates = [0.01, 1]
    batch_sizes = [30, 60, 100, 200]
    hidden_units = [512, 400, 300, 256, 128]

    rates = []
    sizes = []
    units = []

    # for rate in learning_rates:
    #     print('\n↓↓↓  LEARNING RATE: ' + str(rate) + ' ↓↓↓')
    #     rates.append({"rate": rate, "precents": machine(False, list_of_feelings, GRU(list_of_feelings, learning_rate=rate))})
    # print("\nTOTAL RATE PRECENTS:")
    # for rate in rates:
    #     print("rate: " + str(rate["rate"]) + " - " + str(rate["precents"]) + "%")
    #
    for rate in learning_rates:
        print('\n↓↓↓  LEARNING RATE: ' + str(rate) + ' ↓↓↓')
        for size in batch_sizes:
            print('\n↓↓↓ SIZES: ' + str(size) + ' ↓↓↓')
            sizes.append(
                {"size": size, "precents": machine(False, list_of_feelings, GRU(list_of_feelings, learning_rate=rate), batch_size=size)})
        print("\nTOTAL SIZES PERCENTS")
        for size in sizes:
            print("size: " + str(size["size"]) + " - " + str(size["precents"]) + "%")

    for unit in hidden_units:
       print('\n↓↓↓ HIDDEN_UNITS: ' + str(unit) + ' ↓↓↓')
       units.append({"unit": unit, "precents": machine(False, list_of_feelings, GRU(list_of_feelings, hidden_units=unit))})
    print("\nTOTAL HIDDEN UNITS PRECENTS:")
    for unit in units:
        print("unit: " + str(unit["unit"]) + " - " + str(unit["precents"]) + "%")

def separate_dataset_to_batches(dataset, batch_size):
    batches = []
    for i in range(0, len(dataset), batch_size):
        batches.append(dataset[i:i + batch_size])
    return batches


def machine(answer, list_of_feelings, architecture, batch_size=100):
    ml = NeuralNetwork(architecture)

    while True:
        if answer:
            dataset_path, examples = Dataset.save_dataset(ArchitectureType.LSTM, int(EXAMPLES/NUMBER_OF_EXAMPLES_IN_BATCH), EXAMPLES,
                                                          'data.npy',
                                                          list_of_feelings)
            with open('list.json', 'w') as f:
                json.dump(list_of_feelings, f)
        else:
            examples = np.load('./all-datasets/30k-happy-sadness-anger/data.npy', allow_pickle=True)

        examples = separate_dataset_to_batches(examples[0], batch_size)

        print("Hello! 😀 I'm PsychoBot.\nMy thing is sentiment analysis.\n")

        accuracy_test = ml.train(examples[:int(len(examples) * TRAINING_SET_PERCENTAGE)], examples[int(len(examples) * TRAINING_SET_PERCENTAGE):], EPOCHS)
        return 100*accuracy_test[-1]

        # ["anger", "disgust", "fear", "joy", "sadness"]
        # anger, disgust, fear, joy, sadness
        # happy, anger, sadness
        # admiration
        # amusement
        # anger
        # annoyance
        # approval
        # caring
        # confusion
        # curiosity
        # desire
        # disappointment
        # disapproval
        # disgust
        # embarrassment
        # enthusiasm
        # fear
        # gratitude
        # grief
        # happy
        # love
        # worry
        # optimism
        # pride
        # realization
        # relief
        # remorse
        # sadness
        # surprise
        # neutral

def get_model():
    # use pre-trained model and use it
    model = downloader.load('glove-twitter-25')

    return model


def main():
    print('\033[1m' + "PsychoBot POC 1.3" + '\033[0m' + "\nAll rights reserved © PsychoBot 2022\n")
    choice = 0
    ml = None
    while choice != 3:
        print("The battle for the Throne of AI awaits, LSTM or OldGRU, which memory will you choose to lead your army?")
        arch = input()
        print(
            "1 - Train the machine on prepared dataset\n2 - Train the machine on un-prepared dataset\n3 - Use working machine\n4 - I've seen enough")
        choice = int(input())
        if choice == 1:
            if ml is not None:
                print(
                    "Training a new machine will overwrite the previous machine you've made.\nAre you sure you want to train a new one? Y/n")
                choice = input()
                print("")
                if choice.lower() == "y":
                    with open('list.json', 'r') as f:
                        list_of_feelings = json.load(f)
                    ml = machine(False, list_of_feelings)
            else:
                with open('list.json', 'r') as f:
                    list_of_feelings = json.load(f)
                if arch == "LSTM":
                    ml = machine(False, list_of_feelings, NEW_LSTM(list_of_feelings))
                if arch == "GRU":
                    ml = machine(False, list_of_feelings, GRU(list_of_feelings))
        elif choice == 2:
            print("Enter the list of feelings:")
            list_of_feelings = input()
            list_of_feelings = list(list_of_feelings.split(", "))
            print(list_of_feelings)
            ml = machine(True, list_of_feelings)
        elif choice == 3:
            print("This is not possible yet")
        elif choice == 4:
            print("Bye bye 👋")
        elif choice == 5:
            print("You enter the secret option")
            with open('list.json', 'r') as f:
                list_of_feelings = json.load(f)
            ml = machine_with_params(list_of_feelings)


if __name__ == '__main__':
    main()
