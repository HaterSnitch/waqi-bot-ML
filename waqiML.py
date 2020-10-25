# This is a sample Python script.
import discord


import nltk
import numpy as np
import statistics
from statistics import mean
from scipy import spatial

import spacy
nlp = spacy.load("en_core_web_md")
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')
import erasehate as eh
from erasehate.classifier import classifier
from erasehate.reclass import reclassboiler_HTML,parse_reclass_form
#from erasehate.reclass import submit_reclassed
from erasehate.submission import reclass_submission
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

def analysis():
    sid = SentimentIntensityAnalyzer()
    comments = ["islam is the worst religion"]
    for comment in comments:
        sentiment = sid.polarity_scores(comment)
    print(sentiment)
    print(word_tokenize(comment))


def similarity_vector(file ):
    comment = "Islam is the worst religion"
    hate_words=comment.split()
    sw = stopwords.words('english')

    with open(file, "r") as file:
         find_verse=file.read()

    l=[]
    hater_comment = nlp(comment)
    find_the_verse = nlp(find_verse)

    return hater_comment.similarity(find_the_verse)


def which_verse():
    sim = []
    comment = "Islam is the worst religion"
    sim.append("women_and_hijab.txt", comment)
    sim.append("men_and_marriage.txt", comment)
    sim.append("islam_reference.txt", comment)
#highest average is verse to find
if __name__ == '__main__':
    analysis()
    similarity_vector("islam_reference.txt")
    #similarity_vector()


