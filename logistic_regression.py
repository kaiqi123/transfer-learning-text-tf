from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
import json

from sklearn import datasets

import nltk
import re
from pandas import DataFrame

#nltk.download('punkt')
gossipcop_data = pd.read_csv('.\data\gossipcop_content_no_ignore.tsv', sep='\t')
politifact_data = pd.read_csv('.\data\politifact_content_no_ignore.tsv',sep = '\t')

# preprocess gos
def gos():
    gossipcop_words = []
    #fo = open("C:\workspace\PycharmProject\\472Project2_logistic_regression\data\gossipcop_words.json",
           #   "w",encoding="utf-8")
    bow = {}
    for content in gossipcop_data['content']:
        sequence = text.text_to_word_sequence(content)

        #print(sequence)
        for key in sequence:
            if key in bow:
                bow[key] = bow.get(key) + 1
            else:
                bow[key] = 1

    with open("../data/gossipcop_words.json", "w") as fo:
        json.dump(bow, fo)

    print('finished')

# preprocess pol
def pol():
    bow = {}
    for content in politifact_data['content']:
        sequence = text.text_to_word_sequence(content)

        #print(sequence)
        for key in sequence:
            if key in bow:
                bow[key] = bow.get(key) + 1
            else:
                bow[key] = 1

    with open("../data/politifact_words.json", "w") as fo:
        json.dump(bow, fo)

    print('finished')


def tfidf(data_set,data_set1):
    sequence = []
    for content in data_set['content']:
        sequence.append(content)

    for content in data_set1['content']:
        sequence.append(content)


    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sequence)
    feature_name = vectorizer.get_feature_names()
    print(X)
    print(X.shape)

    return X, feature_name


def train(data,name):
    '''
    with open("../data/gossipcop_words.json", 'r') as load_f:
        gossipcop = json.load(load_f)
    with open("../data/politifact_words.json", 'r') as load_f:
        politifact = json.load(load_f)
    train_feature = []
    for key in gossipcop:
        go = [gossipcop[key]]
        train_feature.append(go)

    for key in politifact:
        po = [politifact[key]]
        train_feature.append(po)
    print(len(train_feature))
'''
    train_feature = data
    print('feature',train_feature.shape)

    #print('feature',data1.shape[0],data2.shape[0])
    train_labels = np.concatenate((np.zeros(5816), np.ones(415)))
    print('lables',len(train_labels))

    x_train, x_test, y_train, y_test = train_test_split(train_feature, train_labels, test_size=0.33, random_state=10)

    #print(x_train)
    #print(y_train)


    classifier = LogisticRegression(C=0.1, solver='sag')
    classifier.fit(x_train,y_train)
    coefs = classifier.coef_[0]
    print('---coef----',classifier.coef_)
    top_three = np.argpartition(coefs, -3)[:500]
    feature_name = []
    for i in top_three:
        feature_name.append(name[i])
    print('name',feature_name)
    predicted = classifier.predict(x_test)

    print('accuracy_score: %0.5f' % (metrics.accuracy_score(y_test, predicted)))

if __name__ == "__main__":
    data_set, feature_name = tfidf(gossipcop_data,politifact_data)
    print('----',data_set)
    train(data_set,feature_name)