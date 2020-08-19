from vectorizers import GloveVectorizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Loading data
train = pd.read_csv('data_files/deepnlp_classification_data/r8-train-all-terms.txt', header=None, sep='\t')
test = pd.read_csv('data_files/deepnlp_classification_data/r8-test-all-terms.txt', header=None, sep='\t')
train.columns = ['label', 'document']
test.columns = ['label', 'document']

print(train.shape)

vectorizer_glove = GloveVectorizer()
X_train = vectorizer_glove.fit_transform(train.document)
y_train = train.label

X_test = vectorizer_glove.transform(test.document)
y_test = test.label

print(X_train.shape)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)
print("train score:", model.score(X_train, y_train))
print("test score:", model.score(X_test, y_test))
