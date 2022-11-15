from gensim import models, similarities, downloader
import logging
import tempfile

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

Word2Vec = models.word2vec.Word2Vec

# load corpus and train it
# corpus = downloader.load('text8')
# model = Word2Vec(corpus)


# use pre-trained model and use it
model = downloader.load('glove-twitter-100')

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

text = "yo man i was in school today"

arr = text.split(" ")

#for word in arr:
	#print(model[word])
	# print(model.most_similar(word))

king = model["king"]
man = model["man"]
woman = model["woman"]

newQueen = []

for i in range(len(king)):
	newQueen.append(king[i] - man[i] + woman[i])

queen = model["queen"]
print("King:")
print(king)
print("New:")
print(newQueen)
print("Queen:")
print(queen)

# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#online-training-resuming-training
	
# Save & Load our made models:
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#storing-and-loading-models
