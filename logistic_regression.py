from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

gossipcop_data = pd.read_csv('.\data\gossipcop_content_no_ignore.tsv', sep='\t')
politifact_data = pd.read_csv('.\data\politifact_content_no_ignore.tsv',sep = '\t')


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
    train_feature = data
    print('feature',train_feature.shape)

    train_labels = np.concatenate((np.zeros(5816), np.ones(415)))
    print('lables',len(train_labels))

    x_train, x_test, y_train, y_test = train_test_split(train_feature, train_labels, test_size=0.33, random_state=10)

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