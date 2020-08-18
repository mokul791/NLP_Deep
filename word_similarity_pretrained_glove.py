from __future__ import print_function, division
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


#Finding similar words
def find_similar_word(word, n=10):
	"""The function finds 10 similar words from the 10 closest neighbors

	For example, when a word such king is given, it prints similar words

	Args:
		word: input string
		n: number of similar words requested

	Returns:
		Prints mentioned number of neighboring words
	"""
	word = word.lower()
	if word not in word2vec:
		print("%s is not in the glove dictionary" % word)
		return

	v = word2vec[word]

	distance = pairwise_distances(v.reshape(1, D), embedding, metric='cosine').reshape(N)
	# distance metric can be changed to euclidean
	indexes = distance.argsort()[1: n+1]

	print('The similar words for %s are: ' % word.title())

	for idx in indexes:
		neigh_word = idx2word[idx]
		print('\t %s' % neigh_word.title())


# Loading the glove pre-trained word vectors
print('Loading the word vectors from the pre-trained glove vectors...')
print('\n')

word2vec = {}  # dictionary to hold all the word vectors, with word as key and value as vector
embedding = []  # list to store the whole vocabulary vectors
idx2word = []  # list to hold the index to word

with open('data_files/glove.6B/glove.6B.50d.txt', encoding='utf-8') as f:
	for line in f:
		wordvec = line.split()
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

find_similar_word('King')
find_similar_word('project')
find_similar_word('data')
find_similar_word('scientist')
find_similar_word('manager')
find_similar_word('Job')
