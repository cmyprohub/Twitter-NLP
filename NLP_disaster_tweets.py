# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
sub_df = pd.read_csv('./sample_submission.csv')



#identify and drop all duplicate rows
#train_nodup_df = train_df.drop_duplicates() #no dups

trainX_df = pd.DataFrame(train_df['text'])
trainY_df = pd.DataFrame(train_df['target'])

testX_df = pd.DataFrame(test_df['text'])


import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

#lemmatization function

def lemmatize(sentence):
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    lemmatized_sentence = []
    for word, tag in tagged:      
        lemmatized_sentence.append(lemmatizer.lemmatize(word))
    return ' '.join(lemmatized_sentence)

#text processing
# convert to Lowercase, tokenize, remove stopwords,
#special characters, punctuations, numbers
stop_words = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()





# function to convert nltk tag to wordnet tag
def get_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
#tokenize the sentence and find the POS tag for each token
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    lemmatized_sentence = []
    for word, tag in tagged:
        wntag = get_wordnet_tag(tag)
        if wntag is None:
              lemmatized_sentence.append(lemmatizer.lemmatize(word))
        else:
              lemmatized_sentence.append(lemmatizer.lemmatize(word, pos = wntag))
    return ' '.join(lemmatized_sentence)

import re
trainX_df['text']  = trainX_df['text'].apply(lambda x: re.sub(r'[^\x00-\x7f]',r'', x))
trainX_df['text']  = trainX_df['text'].apply(lambda x: re.sub(r"http\S+", "", x))
trainX_df['text'] = trainX_df['text'].str.replace('[0-9]+', '')
trainX_df['text'] = trainX_df['text'].apply(lambda x: x.lower())
trainX_df['text'] = trainX_df['text'].str.replace('[^\w\s]', '')
trainX_df['text'] = trainX_df['text'].apply(lambda x: ' '.join(w for w in x.split() if w not in stop_words))
trainX_df['text'] = trainX_df['text'].apply(lambda x: lemmatize_sentence(x))

testX_df['text']  = testX_df['text'].apply(lambda x: re.sub(r'[^\x00-\x7f]',r'', x))
testX_df['text']  = testX_df['text'].apply(lambda x: re.sub(r"http\S+", "", x))
testX_df['text'] = testX_df['text'].str.replace('[0-9]+', '')
testX_df['text'] = testX_df['text'].apply(lambda x: x.lower())
testX_df['text'] = testX_df['text'].str.replace('[^\w\s]', '')
testX_df['text'] = testX_df['text'].apply(lambda x: ' '.join(w for w in x.split() if w not in stop_words))
testX_df['text'] = testX_df['text'].apply(lambda x: lemmatize_sentence(x))

#tf-idf vectorization after text processing 
#to ceate text vectors

from sklearn.feature_extraction.text import TfidfVectorizer

tvec = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
trainX_tfidf = tvec.fit_transform(trainX_df['text'])
testX_tfidf = tvec.transform(testX_df['text'])


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(trainX_tfidf, trainY_df)
svpredictions = svclassifier.predict(testX_tfidf)


sub_df['target'] = svpredictions.astype(int)
sub_df.to_csv('submission.csv', index=False)


#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler(with_mean=False)
#X_train_sc = sc.fit(trainX_tfidf)
#trainX_tfidf_sc = X_train_sc.transform(trainX_tfidf)
#testX_tfidf_sc = X_train_sc.transform(testX_tfidf)

#from sklearn.decomposition import PCA
#trainX_pca_df = PCA().fit(trainX_tfidf_sc)

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(trainX_tfidf, trainY_df, random_state=1)

#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#classifier = AdaBoostClassifier(
#    DecisionTreeClassifier(max_depth=1),
#    n_estimators=240
#)
#classifier.fit(train_X, train_y)
#predictions = classifier.predict(test_X)
#
#from sklearn.metrics import confusion_matrix
#confusion_matrix(test_y, predictions)
#
#from sklearn.metrics import accuracy_score
#accuracy_score(test_y, predictions)
#
#from sklearn.metrics import f1_score
#f1_score(test_y, predictions)
#
#trainX_df.head(967)
#
#testX_df.head(6)
#
##Import Gaussian Naive Bayes model
#from sklearn.naive_bayes import MultinomialNB 
#
##Create a Gaussian Classifier
#model = MultinomialNB ()
#
## Train the model using the training sets
#model.fit(train_X.todense(), train_y)
#npredictions = model.predict(test_X.todense())
#
#from sklearn.metrics import accuracy_score
#accuracy_score(test_y, npredictions)
#
#from sklearn.metrics import f1_score
#f1_score(test_y, npredictions)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(train_X, train_y)
svpredictions = svclassifier.predict(test_X)

from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, svpredictions)

from sklearn.metrics import accuracy_score
accuracy_score(test_y, svpredictions)

from sklearn.metrics import f1_score
f1_score(test_y, svpredictions)

sub_df['target'] = svpredictions.astype(int)
sub_df.to_csv('submission.csv', index=False)



from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
classifier = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=240
)
classifier.fit(trainX_tfidf, trainY_df)
predictions = classifier.predict(testX_tfidf)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(trainX_tfidf, trainY_df)
svpredictions = svclassifier.predict(testX_tfidf)



sub_df['target'] = svpredictions.astype(int)
sub_df.to_csv('submission.csv', index=False)