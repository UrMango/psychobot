import pandas as pd
from gensim import models, similarities, downloader
import logging
import re
import spacy

class Dataset:
	df = pd.read_csv(r'E:\GitHub\hadera-801-psychobot\Dataset\go_emotions_dataset.csv')
	last_count = 0
	nlp = spacy.load("en_core_web_sm")

	@staticmethod
	def make_examples(num_of_batches, examples_per_batch):  # [[[][]][][]]
		model = Dataset.get_model()

		vector = []
		for i in range(num_of_batches):
			print("BATCH:", i, "/", num_of_batches)
			vector.append([])
			inner_vector = vector[i]
			for j in range(examples_per_batch):
				inner_vector.append([])
				inner_vector2 = inner_vector[j]

				while (
					Dataset.df.example_very_unclear[Dataset.last_count] == 'TRUE' and
					Dataset.df.anger[Dataset.last_count] == 0
					and Dataset.df.disgust[Dataset.last_count] == 0
					and Dataset.df.fear[Dataset.last_count] == 0
					and Dataset.df.joy[Dataset.last_count] == 0
					and Dataset.df.sadness[Dataset.last_count] == 0
					and Dataset.df.neutral[Dataset.last_count] == 0):
					Dataset.last_count += 1

				# sentence
				text = Dataset.df.text[Dataset.last_count]

				regex = re.compile(r'[^a-zA-Z\s]')
				text = regex.sub('', text)
				text = text.lower()

				# sentence => array of words
				arr = text.split(" ")

				we_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
				res = []

				# words => word embedding
				for word in arr:
					doc = Dataset.nlp(word)
					if doc[0].is_stop:
						continue

					try:
						word_vec = model[word]
						for k in range(len(word_vec)):
							we_arr[k] += word_vec[k]
					except Exception:
						print(word + " wasn't found on word embedding.")

				# 28 feelings vector res
				res.append(Dataset.df.anger[Dataset.last_count])
				res.append(Dataset.df.disgust[Dataset.last_count])
				res.append(Dataset.df.fear[Dataset.last_count])
				res.append(Dataset.df.joy[Dataset.last_count])
				res.append(Dataset.df.sadness[Dataset.last_count])
				res.append(Dataset.df.neutral[Dataset.last_count])

				arr_final = []

				inner_vector2.append([we_arr])
				inner_vector2.append([res])

		return vector

	@staticmethod
	def get_model():

		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

		# use pre-trained model and use it
		model = downloader.load('glove-twitter-25')

		return model