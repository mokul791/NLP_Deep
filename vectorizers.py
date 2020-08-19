import numpy as np
from gensim.models import KeyedVectors


# Define a Glove vectorizer class with fit, transform and fit_transform methods
class GloveVectorizer:
	"""Converts a collection of raw documents to a matrix of GolVe word embeddings
	"""
	def __init__(self):
		# Load the pre-trained Golve word embedding
		print('Loading Glove word vectors')

		word2vec = {}  # dictionary to hold all the word vectors, with word as key and value as vector
		embedding = []  # list to store the whole vocabulary vectors
		idx2word = []  # list to hold the index to word

		with open('data_files/glove.6B/glove.6B.50d.txt', encoding='utf-8') as f:
			for line in f:
				wordvec = line.split()  # The txt contents are space separated
				word = wordvec[0]
				vec = np.asarray(wordvec[1:], dtype='float32')
				word2vec[word] = vec
				embedding.append(vec)
				idx2word.append(word)
		print('The size of vocabulary is %s' % len(word2vec))
		print('\n')

		self.wor2vec = word2vec
		self.embedding = np.array(embedding)
		self.wor2index = {v: k for k, v in enumerate(idx2word)}
		self.N, self.D = self.embedding.shape

	def fit(self, data):
		pass

	def transform(self, data):
		X = np.zeros((len(data), self.D))
		n = 0
		count_empty = 0
		for sentence in data:
			sent = sentence.lower().split()
			vectors = []
			for word in sent:
				if word in self.wor2vec:
					vec = self.wor2vec[word]
					vectors.append(vec)
			if len(vectors) > 0:
				vectors = np.array(vectors)
				X[n] = vectors.mean(axis=0)
			else:
				count_empty += 1
			n += 1
		print("The number of words not found in Glove vectors is %s" % count_empty)
		return X

	def fit_transform(self, data):
		"""Fit to data and then transform it

		Args:
			data: text data

		Returns:
			Transformed data
		"""
		self.fit(data)
		return self.transform(data)








