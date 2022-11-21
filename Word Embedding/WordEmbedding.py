from gensim import models, similarities, downloader
import logging
import tempfile

"""
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

Word2Vec = models.word2vec.Word2Vec
"""

# load corpus and train it
# corpus = downloader.load('text8')
# model = Word2Vec(corpus)

class WordEmbedding:
	@staticmethod
	def get_model():
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

		# use pre-trained model and use it
		model = downloader.load('glove-twitter-100')

		return model

"""
with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
    temporary_filepath = tmp.name
    model.save(temporary_filepath)
    #
    # The model is now safely stored in the filepath.
    # You can copy it to other machines, share it with others, etc.
"""

"""
temporary_filepath = "c:\\temp\\gensim-model-o0u8bur7"
model = models.keyedvectors.KeyedVectors.load(temporary_filepath)
"""

#text = "yo man i was in school today"

#arr = text.split(" ")

#for word in arr:
#	print(model[word])

# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#online-training-resuming-training
	
# Save & Load our made models:
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#storing-and-loading-models
