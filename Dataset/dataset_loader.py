import pandas as pd
from gensim import models, similarities, downloader
import logging
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

df = pd.read_csv(r'go_emotions_dataset.csv')

last_count = 0

def make_examples(num_of_batches, examples_per_batch):  # [[[][]][][]]
	global last_count
	global df

	model = get_model()

	vector = []
	for i in range(num_of_batches):
		vector.append([])
		inner_vector = vector[i]
		for j in range(examples_per_batch):
			inner_vector.append([])
			inner_vector2 = inner_vector[j]

			while df.example_very_unclear[last_count] == 'TRUE':
				last_count += 1

			# sentence
			text = df.text[last_count]

			regex = re.compile(r'[^a-zA-Z\s]')
			text = regex.sub('', text)
			text = text.lower()

			# sentence => array of words
			arr = text.split(" ")

			we_arr = []
			res = []

			# words => word embedding
			for word in arr:
				# TODO: try
				word_vec = model[word]
				for value in word_vec:
					we_arr.append(value)

			# 28 feelings vector res
			res.append(df.admiration[last_count])
			res.append(df.amusement[last_count])
			res.append(df.anger[last_count])
			res.append(df.annoyance[last_count])
			res.append(df.approval[last_count])
			res.append(df.caring[last_count])
			res.append(df.confusion[last_count])
			res.append(df.curiosity[last_count])
			res.append(df.desire[last_count])
			res.append(df.disappointment[last_count])
			res.append(df.disapproval[last_count])
			res.append(df.disgust[last_count])
			res.append(df.embarrassment[last_count])
			res.append(df.excitement[last_count])
			res.append(df.fear[last_count])
			res.append(df.gratitude[last_count])
			res.append(df.grief[last_count])
			res.append(df.joy[last_count])
			res.append(df.love[last_count])
			res.append(df.nervousness[last_count])
			res.append(df.optimism[last_count])
			res.append(df.pride[last_count])
			res.append(df.realization[last_count])
			res.append(df.relief[last_count])
			res.append(df.remorse[last_count])
			res.append(df.sadness[last_count])
			res.append(df.surprise[last_count])
			res.append(df.neutral[last_count])

			inner_vector2.append([we_arr])
			inner_vector2.append([res])

	return vector

def get_model():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	# use pre-trained model and use it
	model = downloader.load('glove-twitter-25')

	return model


if __name__ == '__main__':
	print(make_examples(2, 3))