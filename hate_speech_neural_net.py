from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np
import json
import statistics
from statistics import mean
from scipy import spatial
import tensorflow as tf
from tensorflow.keras.layers import LSTM

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
dimension=10000
INPUT_LENGTH=15;
embed_matrix=[[]]
tokenizer=Tokenizer(num_words=dimension, lower=True)
def clean():
    #tokenizer=Tokenizer(num_words=dimension, lower=True)
    line=[]
    file="x_train_clean.csv"
    with open(file, "r") as file:
         xtrain=file.read()

    tokenizer.fit_on_texts(xtrain)
    tokenizer.texts_to_sequences(xtrain)

    train_index=[line for line in xtrain]
    nltk.pad_sequences(train_index, maxlen=15)

    embeddings= dict()

    test=open("C:\\Users\mayan\PycharmProjects\waqiMLTechnica2020\glove.twitter.27B.200D", encoding='utf8')

    for line in test:
            vals=line.split()
            word=vals[0]
            coefs=np.asarray(vals[1: ], dtype= 'float')
            embeddings[word]=coefs
    test.close()
def Embedding(param, output_dimensions, embed_initialize, trainable):
    embed_matrix = np.zeros((dimension,output_dimensions))
    for word, ind in tokenizer.word_index.items():
        if ind >=dimension:
            continue
        else:
            embed_vect=keras.embeddings.get(word)
            if embed_vect is not None:
                embed_matrix[ind]=embed_vect



def model():
    input_layer = keras.Input(shape=(INPUT_LENGTH,))
    embedding= Embedding(dimension + 1, output_dimensions=200,
                         embed_initialize=tf.keras.initializers.Constant(embed_matrix),
                         trainable=False)
    lstnet1=LSTM(units=10, return_sequences=False)

    drop_out: object=keras.Dropout(0.2)(lstnet1)
    dense=keras.Dense(output_dim=3, init='unifomrm', activation= 'relu')(drop_out)
    output=keras.Dense(output_dim=1, init='uniform', activation= 'sigmoid')(dense)

    model= keras.Model(inputs=input_layer, outputs=output)
    return model
classifier=model()
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
classifier.summary()

class_weight_dict={0: 1.5, 1: 1}

history_tweet= classifier(xtrainPadding, ytrain, validation_data=(x_testPadding, ytest),
                          class_weight=class_weight_dict, batch_size=100, nb_epoch=10)

def evaluate(message):
    test=message

    with open("C:\\Users\mayan\PycharmProjects\waqiMLTechnica2020\\x_train_clean.csv", 'r') as f:
         dictionary=json.load(f)

    message_index=[]
    for word in message.split():
        if word in dictionary.keys():
            message_index.append(dictionary.get(word))
    for num in message_index:
        if num>dimension:
            message_index.remove(message_index[message_index.index(num)])
    message_indexPad=keras.pad_sequences([message_index], maxlen=INPUT_LENGTH)

    prediction=classifier.predict(message_indexPad)
    print('the probability of being offensive: {0: .2f}'.format(prediction[0][0]))




