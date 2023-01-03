import numpy as np
import pandas as pd
from gensim import models, similarities, downloader
import logging
import re
import random
import spacy
from NeuralNetwork.Architectures.Architecture import ArchitectureType


class Dataset:
    df = pd.read_csv(
        r'E:\GitHub\hadera-801-psychobot\Dataset\go_emotions_dataset.csv')
    last_count = 0
    nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def make_examples(architecture, num_of_batches, examples_per_batch, list_of_feelings):
        """
		Makes examples by a specified number of batches and examples per batch
		Number:param num_of_batches:
		Number:param examples_per_batch:
		List:param list_of_feelings: list of emotions names to learn
		:return:
		"""
        model = Dataset.get_model()

        emotions_count = []
        for i in range(len(list_of_feelings)):
            emotions_count.append(0)
        # anger = 0
        # disgust = 0
        # fear = 0
        # joy = 0
        # sadness = 0
        # neutral = 0
        vector = []

        try:
            for i in range(num_of_batches):
                print("BATCH:", i, "/", num_of_batches)
                vector.append([])
                inner_vector = vector[i]
                for j in range(examples_per_batch):
                    inner_vector2 = []

                    stop_grow = False

                    for k, emotion in enumerate(list_of_feelings):
                        # Check if the value of the current emotion column at the last count index is 1
                        # and if the current emotion count is greater than (examples_per_batch*num_of_batches / 5)
                        if (getattr(Dataset.df, emotion)[Dataset.last_count] == 1 and emotions_count[k] - 1 > (
                                examples_per_batch * num_of_batches / len(list_of_feelings))):
                            stop_grow = True
                            break

                    while (Dataset.df.example_very_unclear[Dataset.last_count] == 'TRUE' or
                           all(getattr(Dataset.df, emotion)[Dataset.last_count] == 0 for emotion in
                               list_of_feelings)) or stop_grow:
                        Dataset.last_count += 1
                        for k, emotion in enumerate(list_of_feelings):
                            # Check if the value of the current emotion column at the last count index is 1
                            # and if the current emotion count is greater than (examples_per_batch*num_of_batches / 5)
                            if (getattr(Dataset.df, emotion)[Dataset.last_count] == 1 and emotions_count[k] - 1 > (
                                    examples_per_batch * num_of_batches / len(list_of_feelings))):
                                stop_grow = True
                                break

                    # sentence
                    text = Dataset.df.text[Dataset.last_count]

                    for k, emotion in enumerate(list_of_feelings):
                        if getattr(Dataset.df, emotion)[Dataset.last_count] != 0:
                            emotions_count[k] += 1

                    # if Dataset.df.neutral[Dataset.last_count] != 0:
                    # neutral += 1

                    regex = re.compile(r'[^a-zA-Z\s]')
                    text = regex.sub('', text)
                    text = text.lower()

                    # sentence => array of words
                    arr = text.split(" ")

                    we_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    words_arr = []
                    res = []

                    # words => word embedding
                    for word in arr:
                        doc = Dataset.nlp(word)
                        if doc and doc[0].is_stop:
                            continue
                        try:
                            # print("Before:", word)
                            word = doc[0].lemma_
                            # print("After:", word)
                            word_vec = model[word]
                            if architecture == ArchitectureType.LSTM:
                                words_arr.append(word_vec)
                                continue
                            for k in range(len(word_vec)):
                                if architecture == ArchitectureType.BASIC:
                                    we_arr[k] += word_vec[k]
                        except Exception:
                            pass
                        # print(word + " wasn't found on word embedding.")

                    if architecture == ArchitectureType.BASIC:
                        for k in range(len(we_arr)):
                            we_arr[k] /= 25

                    # 28 (6) feelings vector res
                    for k, emotion in enumerate(list_of_feelings):
                        res.append(getattr(Dataset.df, emotion)[Dataset.last_count])

                    print('\r' + str(j) + "/" + str(examples_per_batch), end="")

                    arr_final = []
                    if architecture == ArchitectureType.BASIC:
                        if len(we_arr) == 0:
                            Dataset.last_count += 1
                            continue
                        inner_vector2.append(we_arr)
                    elif architecture == ArchitectureType.LSTM:
                        if len(words_arr) <= 1:
                            Dataset.last_count += 1
                            continue
                        inner_vector2.append(words_arr)
                    inner_vector2.append(res)

                    inner_vector.append(inner_vector2.copy())

                    Dataset.last_count += 1
        except Exception as e:
            print(e)

        for batch in vector:
            random.shuffle(batch)
        amount_for_print = ""
        for i in range(len(list_of_feelings)):
            amount_for_print = amount_for_print + str(emotions_count[i])+", "
        print(amount_for_print)

        return vector

    @staticmethod
    def get_model():

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # use pre-trained model and use it
        model = downloader.load('glove-twitter-25')

        return model

    @staticmethod
    def save_dataset(arch_type, batches, examples, file_name, list_of_feelings):
        examples = Dataset.make_examples(arch_type, batches, examples, list_of_feelings)

        np.save(file_name, examples)

        return file_name, examples
