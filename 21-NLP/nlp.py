# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 00:38:58 2022

@author: reco1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

yorumlar = pd.read_csv("Restaurant_Reviews.csv",  on_bad_lines='skip')
#yorumlar = yorumlar.dropna(axis=0)

import re
#(natural langugae toolkit)
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download("stopwords")
from nltk.corpus import stopwords


'''
#"a-zA-Z" bu harfler aralığında olmayan harfleri al boşluk at, noktalama işaretlerini yani     
yorum = re.sub("[^a-zA-Z]", " ", yorumlar["Review"][0]) 
yorum = yorum.lower() # tamamını küçük harfe dönüştürme
yorum = yorum.split() # listeye dönüştürme

#ingilizce stopwordsleri kümeye çevirip, 
#kümenin içerisinde kelime yoksa bu kelimeyi stemle,
#bunu da listenin ilk elemanı yap
yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]   
#sonuc olarak ekleri atıp listeye yazdık

yorum = " ".join(yorum) # yorumu al boşluklarla birleştir
'''

#Preprocessing (Önişleme)
derlem = []

for i in range(len(yorumlar)):
    yorum = re.sub("[^a-zA-Z]", " ", yorumlar["Review"][i]) 
    yorum = yorum.lower() 
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]   
    yorum = " ".join(yorum)
    derlem.append(yorum)

#Feature Extraction (Öznitelik Çıkarımı)
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000) # en fazla kullanılan 2000 kelimeyi al
X = cv.fit_transform(derlem).toarray() # bağımsız değişken
y = yorumlar.iloc[:,1].values # bağımlı değişken

#Makine Öğrenmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import ConfusionMatrix
cm = ConfusionMatrix(y_test, y_pred)
print(cm)


