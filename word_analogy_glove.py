from __future__ import print_function, division
from future.utils import iteritems
# from builtins import range
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


# two distance function will be used: Euclidean and cosine
def dist_euclidean(a, b):
	return np.linalg.norm(a - b)


def dist_cosine(a, b):
	return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Defining word analogy function with naive implementation.
def word_analogy_naive(word1, word2, word3):
	""" Find the nearest word from the Glove  for word representation.

	For example, word analogy will used to find the word from a structure such as
	king - man = unknown_word - woman, where it will find the the word queen.

	Args:
		word1, word2, word3: strings in lower-case.

	Returns:
		the unknown word from the resultant vector.
	"""

	for word in (word1, word2, word3):
		if word not in word2vec:
			print('%s is not in the Glove dictionary' % word)
			break

	x = word2vec[word1]
	y = word2vec[word2]
	z = word2vec[word3]
	v0 = x - y + z

	min_dist = float('inf')
	best_word = ''
	for word, vector in iteritems(word2vec):
		if word not in (word1, word2, word3):
			distance = dist_cosine(v0, vector)  # distance metric either cosine or euclidean
			if distance < min_dist:
				min_dist = distance
				best_word = word

	print('\t', word1, ' - ', word2, ' = ', best_word, ' - ', word3)


# Defining word analogy function with scikit-learn pairwise distance. This algorithm is faster
def word_analogy_faster(word1, word2, word3):
	""" Find the nearest word from the Glove  for word representation with pairwise distance.

	For example, word analogy will used to find the word from a structure such as
	king - man = unknown_word - woman, where it will find the the word queen.

	Args:
		word1, word2, word3: strings in lower-case.

	Returns:
		the unknown word from the resultant vector.
	"""

	for word in (word1, word2, word3):
		if word not in word2vec:
			print('%s is not in the Glove dictionary' % word)
			break

	x = word2vec[word1]
	y = word2vec[word2]
	z = word2vec[word3]
	v0 = x - y + z

	distance = pairwise_distances(v0.reshape(1, D), embedding, metric='cosine').reshape(N)
	indices = distance.argsort()[:4]  # taking first 4 word index
	for idx in indices:
		word = idx2word[idx]
		if word not in (word1, word2, word3):
			best_word_faster = word
			break

	print('\t', word1, ' - ', word2, ' = ', best_word_faster, ' - ', word3)


# Loading the glove pre-trained word vectors
print('Loading the word vectors from the pre-trained glove vectors...')
print('\n')

word2vec = {}  # dictionary to hold all the word vectors, with word as key and value as vector
embedding = []  # list to store the whole vocabulary vectors
idx2word = []  # list to hold the index to word

with open('data_files/glove.6B/glove.6B.50d.txt', encoding='utf-8') as f:
	for line in f:
		wordvec = line.split()  # The txt file is space separated
		word = wordvec[0]
		vec = np.asarray(wordvec[1:], dtype='float32')
		word2vec[word] = vec
		embedding.append(vec)
		idx2word.append(word)

print('The size of vocabulary is %s' % len(word2vec))
print('\n')

embedding = np.array(embedding)
N, D = embedding.shape
print("The shape of embedding is: " + str(N) + ' X ' + str(D))
print('\n')

# testing the word similarity with naive algorithm
print('Result from manual implementation of distance metric: ')
word_analogy_naive('france', 'paris', 'london')
word_analogy_naive('japan', 'japanese', 'chinese')
word_analogy_naive('man', 'woman', 'sister')

# testing the word similarity with naive algorithm
print('\n')
print('Result from scikit-learn pairwise distance: ')
word_analogy_faster('france', 'paris', 'london')
word_analogy_faster('japan', 'japanese', 'chinese')
word_analogy_faster('man', 'woman', 'sister')





