from gensim.models import KeyedVectors


# Load vectors
print('Loading word vectors...')
vectors = KeyedVectors.load_word2vec_format('data_files/GoogleNews-vectors-negative300.bin', binary=True)


def word_analogy(word1, word2, word3):
	""" Find the word analogy from the google word vector representation.

	For example, word analogy will used to find the word from a structure such as
	king - man = unknown_word - woman, where it will find the the word queen.

	Args:
		word1, word2, word3: strings.

	Returns:
		the unknown word from the resultant vector.
	"""
	print('Finding word analogy...')
	word_values = vectors.most_similar(positive=[word1, word3], negative=[word2])
	print("\t%s - %s = %s - %s" % (word1, word2, word_values[0][0], word3))


def find_similar_words(word, n=5):
	"""The function finds 5 similar words from the 10 closest neighbors

		For example, when a word such king is given, it prints similar words

		Args:
			word: input string
			n: number of similar words requested

		Returns:
			Prints mentioned number of neighboring words
		"""

	word_values = vectors.most_similar(positive=[word])
	print('Finding similar words for %s ...' % word)
	for w, val in word_values[:n]:
		print("\t %s " % w)


word_analogy('king', 'man', 'woman')
word_analogy('france', 'paris', 'london')
word_analogy('japan', 'japanese', 'chinese')
word_analogy('man', 'woman', 'sister')

print('\n')

find_similar_words('King', n=10)
find_similar_words('project', n=10)
find_similar_words('data', n=10)
find_similar_words('scientist', n=10)
find_similar_words('manager', n=10)
find_similar_words('Job', n=10)

