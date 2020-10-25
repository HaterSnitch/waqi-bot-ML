import math
from operator import gt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import csv
import random


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
    for i in range(len(classSummaries)):
        mean, stdev = classSummaries[i]
    x = inputVector[i]
    probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def remove_emojis():
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
     if bestLabel is None or probability  &gt: bestProb
     bestProb = probability
     bestLabel = classValue
    return bestLabel
def getPredictions(summaries, testSet):
    predictions=[]
    for i in range(len(testSet)):
        result=predict(summaries, testSet[i])
        predictions.append(result)
    return predictions
def getAccuracy(testSet, predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct+=1
    return (correct/float(len(testSet)))*100.0
def separateByClass(dataset):
    separated={}
    for i in range (len(dataset)):
        vector=dataset[i]
        if(vector[-1] not in separated):
            separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
    return separated
def mean(numbers):
    return sum(numbers)/float(len(numbers))
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated=separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue]= summarize(instances)
    return summaries

def is_it_islamaphobic():

    splitRatio = 0.67
    dataset = np.genfromtxt('noislamophobia-dataset.csv', delimiter=',')
    trainingSet, testSet = np.split(dataset, splitRatio)
    print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))

if __name__ == '__main__':
    is_it_islamaphobic()