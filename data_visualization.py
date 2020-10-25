
import nltk
import numpy as np
import pandas as pd
import statistics
from statistics import mean
from scipy import spatial
from hatesonar import Sonar
import sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
nlp = spacy.load("en_core_web_md")
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')

def data_visualization(df):
    sentiments = {'neg': ' ', 'neut':  ' ', 'pos': ' '}

    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # oject gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.



    for ind in df.index:
        sentiment_dict = sid_obj.polarity_scores(df[ind])
        sentiments['neut']=sentiment_dict['neg'] * 100
        sentiments['neut']=sentiment_dict['neu'] * 100
        sentiments['pos']=sentiment_dict['pos']*100
        print("Overall sentiment dictionary is : ", sentiment_dict)

    print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")

    print("Sentence Overall Rated As", end=" ")
if __name__ == '__main__':
    df = pd.read_csv("noislamophobia-dataset.csv")
    df = df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    data_visualization(df)