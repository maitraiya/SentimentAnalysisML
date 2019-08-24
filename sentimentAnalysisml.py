import tweepy
import imp
import re
import nltk
nltk.download('stopwords')
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from flask import Flask,render_template,request
import os

consumer_key = os.environ['consumer_key_value']
consumer_secret = os.environ['consumer_secret_value']

access_token = os.environ['access_token_value']
access_token_secret = os.environ['access_token_secret_value']
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


from sklearn.externals import joblib
classifier = joblib.load('model')
bow_vectorizer=joblib.load('vocab')

combi =[]

app=Flask(__name__,template_folder='template')
@app.route('/')
def tweets():
    train_tweets=[]
    query = 'Machine Learning'
    public_tweets = api.search(query,count = 100,lang='en',result_type="recent")

    positive=0
    neutral=0
    negative=0
    mydict={}
    tweetlist =[]
    response = ''
    for tweet in public_tweets:
        tweett =re.sub("(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet.text)
        tweett = tweett.lower()
        tweett = tweett.split()
        twe = []
        lem = WordNetLemmatizer()

        for t in tweett:
            if (not (t == 'rt')):
                t = lem.lemmatize(t)
                twe.append(t)
        tweett = ' '.join(twe)
        tweett = tweett.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in tweett if not word in set(stopwords.words('english'))]
        tweett = ' '.join(review)
        train_tweets.append(tweett)
        tweetlist.append(tweet.text)

    from sklearn.feature_extraction.text import CountVectorizer
    z = bow_vectorizer.transform(train_tweets).toarray()
    y_pred = classifier.predict(z)
    i=0
    for tweett in tweetlist:
        mydict.update({tweett:y_pred[i]})
        if y_pred[i]>0:
            positive=positive+1
        else:
            negative = negative+1
        i=i+1

    total=positive+negative
    positive = round((positive/total)*100,2)
    negative = round((negative/total)*100,2)
    return render_template('tweets.html',query=query,mydict=mydict,pos= positive,neg=negative)


@app.route('/tweetsdisplay.html',methods=["POST","GET"])
def tweetsdisplay():
    train_tweets=[]
    query = request.form.get('query')
    public_tweets = api.search(query,count = 100,lang='en',result_type="recent")

    positive=0
    neutral=0
    negative=0
    mydict={}
    tweetlist =[]
    response = ''
    for tweet in public_tweets:
        tweett =re.sub("(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet.text)
        tweett = tweett.lower()
        tweett = tweett.split()
        twe = []
        lem = WordNetLemmatizer()

        for t in tweett:
            if (not (t == 'rt')):
                t = lem.lemmatize(t)
                twe.append(t)
        tweett = ' '.join(twe)
        tweett = tweett.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in tweett if not word in set(stopwords.words('english'))]
        tweett = ' '.join(review)
        train_tweets.append(tweett)
        tweetlist.append(tweet.text)

    from sklearn.feature_extraction.text import CountVectorizer
    z = bow_vectorizer.transform(train_tweets).toarray()
    y_pred = classifier.predict(z)
    i=0
    for tweett in tweetlist:
        mydict.update({tweett:y_pred[i]})
        if y_pred[i]>0:
            positive=positive+1
        else:
            negative = negative+1
        i=i+1

    total=positive+negative
    positive = round((positive/total)*100,2)
    negative = round((negative/total)*100,2)
    return render_template('tweets_display.html',query=query,mydict=mydict,pos= positive,neg=negative)


if __name__ == '__main__':
   app.run(debug = True)
