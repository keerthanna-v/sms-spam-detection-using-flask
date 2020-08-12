# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:41:15 2020

@author: keerthanna
"""

from flask import Flask, render_template, url_for, request
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle


model=pickle.load(open("model.pkl",'rb'))
tf=pickle.load(open("tf.pkl",'rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data1 = message
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(data1)
        words = [word for word in tokens if word.isalpha()]
        words = [word for word in words if not word in stop_words]
        w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        words= [lemmatizer.lemmatize(w) for w in words]
        data1 = " ".join(words)
        vect = tf.transform([data1]).toarray()
        my_prediction = model.predict(vect)
    return render_template('result.html',prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
