# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:22:04 2020

@author: keerthanna
"""

# importing packages



import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from text_processing import puncation,stopwordsremoval,pstem
import pickle

# loading the data set

data=pd.read_csv(r"spam.csv",encoding='latin')

data['text']=data['text'].apply(puncation)

data['text']=data['text'].apply(lambda x: x.lower())
# removal of stopwords

data['text']=data['text'].apply(stopwordsremoval)

#stemming

data['text']=data['text'].apply(pstem)

# splitng the data set

x_train,x_test,y_train,y_test=train_test_split(data['text'], data['type'], test_size=0.2, random_state=3)

tf= TfidfVectorizer()

tfidf_train=tf.fit_transform(x_train)

tfidf_test=tf.transform(x_test)

pac=PassiveAggressiveClassifier()

pac.fit(tfidf_train,y_train)


pickle.dump(pac,open("model.pkl",'wb'))
pac=pickle.load(open('model.pkl','rb'))
pickle.dump(tf,open("tf.pkl",'wb'))
tf=pickle.load(open('tf.pkl','rb'))
tf.transform(["he kja s"])
pac.predict(tfidf_test[0])
