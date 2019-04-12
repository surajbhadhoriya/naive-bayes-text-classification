# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:25:02 2019

@author: SURAJ BHADHORIYA
"""
#load libraries
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB

#load dataset
data=pd.read_csv("amazon_dataset.csv")
#making labels
data=data[data['rating'] !=3]
print(data['rating'])
data['sentimate']=data['rating'].apply(lambda rating:+1 if rating>3 else -1)

feature=["review"]
label=["sentimate"]
#make data frame
df=pd.DataFrame(data[feature+label][:40000])
df=df.fillna({'review':''})
df['sentimate'].mean()

X_train,X_test,y_train,y_test=train_test_split(df['review'],df['sentimate'],test_size=0.3)
#apply regularexpression with bag of words
from nltk.tokenize import RegexpTokenizer
token=RegexpTokenizer(r'[a-zA-Z0-9]+')
cv=CountVectorizer(tokenizer=token.tokenize)

xt=cv.fit_transform(X_train)
xt1=cv.transform(X_test)


clf=MultinomialNB()
clf.fit(xt,y_train)


accuracy=clf.score(xt1,y_test)
print(accuracy)
xx="it is not good"
pre=clf.predict(xx)
print(pre)




from sklearn.feature_extraction.text import TfidfVectorizer
vect= TfidfVectorizer(min_df=5)

xy=vect.fit_transform(X_train)
xy1=vect.transform(X_test)

clf.fit(xy,y_train)
accuracy1=clf.score(xy1,y_test)
print(accuracy1)

print(clf.predict(vect.transform(['it is really bad foy baby health worst things'])))





