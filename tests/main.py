# Help:
# matrix multiplication - np.dot
# initialize array - np.zeros
# initialize random values array - np.random.uniform
import json
import time
import math
import numpy as np
import spacy
import pickle

import wandb

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

EPOCHS = 15

NUMBER_OF_EXAMPLES_IN_BATCH = 100

EXAMPLES = 2000

TRAINING_SET_PERCENTAGE = 0.8

nlp = spacy.load("en_core_web_sm")


def machine_with_params(list_of_feelings):
    learning_rates = [1, 0.1, 0.01]
    batch_sizes = [60, 100]
    hidden_units = [350, 150]

    beta1_values = [0.95, 0.9, 0.85]
    beta2_values = [0.999, 0.99, 0.95]

    rates = []
    sizes = []
    units = []
    dict_params = {}

    # for rate in learning_rates:
    #     print('\n↓↓↓  LEARNING RATE: ' + str(rate) + ' ↓↓↓')
    #     rates.append({"rate": rate, "precents": machine(False, list_of_feelings, GRU(list_of_feelings, learning_rate=rate))})
    # print("\nTOTAL RATE PRECENTS:")
    # for rate in rates:
    #     print("rate: " + str(rate["rate"]) + " - " + str(rate["precents"]) + "%")
    #
    # for rate in learning_rates:
    #     print('\n↓↓↓  LEARNING RATE: ' + str(rate) + ' ↓↓↓')
    #     for size in batch_sizes:
    #         print('\n↓↓↓ SIZES: ' + str(size) + ' ↓↓↓')
    #         accuracy_test = machine(True, list_of_feelings, GRU(list_of_feelings, learning_rate=rate), batch_size=size)
    #         for i in range(EPOCHS):
    #             dict_params["batch size: "+str(size)+", learning rate: "+str(rate)+", epoch: "+str(i)+" - "] = 100*accuracy_test[i]
    #             print("batch size: "+str(size)+", learning rate: "+str(rate)+", epoch: "+str(i)+" - " + str(100*accuracy_test[i]) + "%")
    #     print("\nTOTAL PERCENTS")
    #     for key in dict_params.keys():
    #         print(key + str(dict_params[key]) + "%")

    #    print('\n↓↓↓  LEARNING RATE: ' + str(rate) + ' ↓↓↓')
    #    for unit in hidden_units:
    #        print('\n↓↓↓ HIDDEN: ' + str(unit) + ' ↓↓↓')
    #        accuracy_test = machine(True, list_of_feelings, GRU(list_of_feelings, learning_rate=rate, hidden_units=unit), batch_size=60)
    #        for i in range(EPOCHS):
    #            dict_params[
    #                "hidden: " + str(unit) + ", learning rate: " + str(rate) + ", epoch: " + str(i) + " - "] = 100 * \
    #                                                                                                               accuracy_test[
    #                                                                                                                   i]
    #            print("hidden: " + str(unit) + ", learning rate: " + str(rate) + ", epoch: " + str(i) + " - " + str(
    #                100 * accuracy_test[i]) + "%")
    #    print("\nTOTAL PERCENTS")
    #    for key in dict_params.keys():
    #        print(key + str(dict_params[key]) + "%")

    for rate in learning_rates:
        print('\n↓↓↓  LEARNING RATE: ' + str(rate) + ' ↓↓↓')
        for value1 in beta1_values:
            print('\n↓↓↓ BETA1 VALUE: ' + str(value1) + ' ↓↓↓')
            for value2 in beta2_values:
                accuracy_test = machine(True, list_of_feelings, GRU(list_of_feelings, learning_rate=rate, beta1=value1, beta2=value2), batch_size=60)
                for i in range(EPOCHS):
                    dict_params[
                        "beta1: " + str(value1) + ", beta2: " + str(value2)+", learning rate: " + str(rate) + ", epoch: " + str(i) + " - "] = 100 * accuracy_test[i]
                    print("beta1: " + str(value1) + ", beta2: " + ", learning rate: " + str(rate) + ", epoch: " + str(i) + " - " + str(
                        100 * accuracy_test[i]) + "%")
        print("\nTOTAL PERCENTS")
        for key in dict_params.keys():
            print(key + str(dict_params[key]) + "%")

    # for unit in hidden_units:
    #    print('\n↓↓↓ HIDDEN_UNITS: ' + str(unit) + ' ↓↓↓')
    #    units.append({"unit": unit, "precents": machine(False, list_of_feelings, GRU(list_of_feelings, hidden_units=unit))})
    # print("\nTOTAL HIDDEN UNITS PRECENTS:")
    # for unit in units:
    #     print("unit: " + str(unit["unit"]) + " - " + str(unit["precents"]) + "%")


def separate_dataset_to_batches(dataset, batch_size):
    batches = []
    for i in range(0, len(dataset), batch_size):
        batches.append(dataset[i:i + batch_size])
    return batches


def machine(answer, list_of_feelings, architecture, batch_size=60):
    ml = architecture
    dataset = '30k-happy-sadness-anger'

    while True:
        if answer:
            dataset_path, examples = Dataset.save_dataset(ArchitectureType.LSTM, EXAMPLES,
                                                          'all-datasets/30k-happy-sadness-anger/data.npy', list_of_feelings)
            with open('all-datasets/30k-happy-sadness-anger/list.json', 'w') as f:
                json.dump(list_of_feelings, f)
        else:
            examples = np.load('./all-datasets/' + dataset + '/data.npy', allow_pickle=True)
        examples = separate_dataset_to_batches(examples, batch_size)
        print("Hello! 😀 I'm PsychoBot.\nMy thing is sentiment analysis.\n")
        accuracy_test = ml.train(examples[:int(len(examples) * TRAINING_SET_PERCENTAGE)], examples[int(len(examples) * TRAINING_SET_PERCENTAGE):], batch_size, EPOCHS, dataset)
        print(accuracy_test)
        return accuracy_test


def get_model():
    # use pre-trained model and use it
    model = downloader.load('glove-twitter-25')

    return model


def main():
    print('\033[1m' + "PsychoBot POC 1.3" + '\033[0m' + "\nAll rights reserved © PsychoBot 2022\n")
    choice = 0
    ml = None
    while choice != 3:
        print("The battle for the Throne of AI awaits, LSTM or GRU, which memory will you choose to lead your army?")
        arch = input()
        print(
            "1 - Train the machine on prepared dataset\n2 - Train the machine on un-prepared dataset\n3 - Use working machine\n4 - I've seen enough")
        choice = int(input())
        if choice == 1:
            with open('all-datasets/30k-happy-sadness-anger/list.json', 'r') as f:
                list_of_feelings = json.load(f)
            if arch == "LSTM":
                accuracy = machine(False, list_of_feelings, NEW_LSTM(list_of_feelings))
                print("Accuracy on validation set: " + str(100 * accuracy[-1]) + "%")
            if arch == "GRU":
                accuracy = machine(False, list_of_feelings, GRU(list_of_feelings))
                print("Accuracy on validation set: " + str(100*accuracy[-1]) + "%")

        elif choice == 2:
            print("Enter the list of feelings:")
            list_of_feelings = input()
            list_of_feelings = list(list_of_feelings.split(", "))
            print(list_of_feelings)
            accuracy = machine(True, list_of_feelings, GRU(list_of_feelings))
            print("Accuracy on validation set: " + str(100 * accuracy[-1]) + "%")
        elif choice == 3:
            # model_name = input("What model will you choose? (default: dutiful-microwave-9) ")
            # if model_name == "":
            #     model_name = "dutiful-microwave-9"
            # with open('all-datasets/30k-happy-sadness-anger/list.json', 'r') as f:
            #     list_of_feelings = json.load(f)
            # print("This is now possible!!!")
            # with open("models/" + model_name + "/parameters_"+str(list_of_feelings)+".json", 'rb') as f:
            #     dict_parameters = pickle.load(f)
            wandb.init(job_type="run")
            artifact = wandb.use_artifact('noamr/model-registry/psychobot:latest', type='model')
            artifact_dir = artifact.download("models/latest")

            dict_parameters = None
            list_of_feelings = []
            with open(artifact_dir + '/dataset/list.json', 'r') as f:
                list_of_feelings = json.load(f)
            with open(artifact_dir + "/model/parameters_" + str(list_of_feelings) + ".json", 'rb') as f:
                dict_parameters = pickle.load(f)

            ml = GRU(list_of_feelings, set_parameters=True, parameters=dict_parameters, embed=True)

            # examples = np.load('./all-datasets/30k-happy-sadness-anger/data.npy', allow_pickle=True)
            # print("Accuracy on validation set: "+str(100*ml.test(examples[int(len(examples) * TRAINING_SET_PERCENTAGE):]))+"%")
            sentence = ""
            while sentence != "Stop":
                sentence = input("Write a sentence that you want the machine will check: ")
                feeling, dict = ml.run_model_with_embedding(sentence)
                print("Feeling: "+ feeling)
                for feel in list_of_feelings:
                    print(feel + ": " + str(dict[feel]))
                print("")
                print("")
        elif choice == 4:
            print("Bye bye 👋")
        elif choice == 5:
            print("You enter the secret option")
            with open('all-datasets/30k-happy-sadness-anger/list.json', 'r') as f:
                list_of_feelings = json.load(f)
            machine_with_params(list_of_feelings)


if __name__ == '__main__':
    main()
