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

consumer_key = 'xyUBSI0D0JUlMjp3XziEcxNlv'
consumer_secret = 'qUM4ZnFRWcdreStbaNY5VebhhbwAo8vWiyo1DlpSBT1FczJiTM'

access_token = '839828874959650817-nbPlOHAE2Jc2R6Akv1RWgNbR0HFqfFy'
access_token_secret = 'oAnRvKfTBo7KjSO4xTAag5GsJOGwVEiEZXnejyWtVQe5K'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# import RESTAURANT_REVIEWS dataset by using pandas
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t',quoting =3)
#saving the reviews in x variable
x = dataset.iloc[:, 0]
#saving the sentiments in y variable
y = dataset.iloc[:, 1]

corpus = []
#loop for reading reviews and clean them
for i in range(0, 1000):
    #removing the elements and symbols which are listed in re.sub() by taking x as input and converting it in string
    review = re.sub("(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", str(x[i]))
    #lower casing the review
    review = review.lower()
    #spliting the review in tokens
    review = review.split()
    #stemming the words in review separately
    ps = PorterStemmer()
    #stemming words and removing stopwords
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #joining the words and again forming the review in sentence
    review = ' '.join(review)
    #storing all the review in corpus list
    corpus.append(review)

#forming bag of words model
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_features=None, stop_words='english')
# bag-of-words feature matrix
#converting each word in vector form for calculation of sentiments
x = bow_vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#Building the model with gaussianNB
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
#training the model by providing x as review and y as sentiments
classifier.fit(x,y)

combi =[]

app=Flask(__name__,template_folder='template')
@app.route('/')
def tweets():
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
        tweetlist.append(tweet.text)

    from sklearn.feature_extraction.text import CountVectorizer
    z = bow_vectorizer.transform(tweetlist).toarray()
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
        tweetlist.append(tweet.text)

    from sklearn.feature_extraction.text import CountVectorizer
    z = bow_vectorizer.transform(tweetlist).toarray()
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
