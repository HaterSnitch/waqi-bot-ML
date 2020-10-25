from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

import nltk
# nltk.download()
from nltk.stem import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import numpy as np
import pandas as pd

df = pd.read_csv('noislamophobia-dataset.csv', sep=',', header=None, names=['label', 'message'], dtype='unicode')
df['message'] = df.message.map(lambda x: x)
df = df.message.str.replace('[^\w\s]', '')

df['message'] = df['message'].apply(nltk.word_tokenize)
stemmer = PorterStemmer()
df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])

df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()
counts = count_vect.fit_transform(df)
transformer = TfidfTransformer().fit(counts)
counts = transformer.transform(counts)

X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=69)
model = MultinomialNB().fit(X_train, y_train)

predicted = model.predict(X_test)
print(np.mean(predicted == y_test))

print(confusion_matrix(y_test, predicted))
