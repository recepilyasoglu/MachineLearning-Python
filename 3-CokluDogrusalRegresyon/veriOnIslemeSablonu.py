# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 03:02:41 2022

@author: reco1
"""
# -*- coding: utf-8 -*-

#1.kutuphaneler
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#2.veri onisleme
#2.1.veri yukleme
data = pd.read_csv("veriler.csv")
# data = pd.read_csv("veriler.csv")

print(data)

#encoder: Kategorik -> Numeric 
ulke = data.iloc[:,1:4].values
print(ulke)

from sklearn import preprocessing

#LabelEncoding
le = preprocessing.LabelEncoder()

# ilk kolonu alıyoruz, bu kolonu transform ediyoruz  
ulke[:,0] = le.fit_transform(data.iloc[:,0]) 
print(ulke)

#encoder: Kategorik -> Numeric 
c = data.iloc[:,-1:].values
print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(data.iloc[:,-1]) 
print(c)


#OneHotEncoding
ohe = preprocessing.OneHotEncoder()

# bi onceki asamada sayıya ulke kolonumuz ogrenilip bunu transform edecek
# 3 kolnu sırasıyla 0 ve 1 lerden oluscak sekilde OneHotEncode'a donusturecek
# toarray le numpy array olarak sonucu alıyoruz
c = ohe.fit_transform(c).toarray()
print(c)


#numpy dizileri dataframe donusumu
# input edilmis hali
sonucUlke = pd.DataFrame(data=ulke, index = range(22), columns = ["fr","tr","us"])
print(sonucUlke)

#sonucYas = pd.DataFrame(data=Yas, index = range(22), columns = ["boy","kilo","yas"])
#print(sonucYas)

# cinsiyet kolonunun degerlerini alıyoruz 
cinsiyet = data.iloc[:,-1].values
print(cinsiyet)

sonucCinsiyet = pd.DataFrame(data = c[:,:1], index = range(22), columns=["cinsiyet"]) 
print(sonucCinsiyet)

#dataframe birlestirme islemi
# axis=1 le yanyana eşleme yapıyor, 0 da alt alta yazıyor
#s = pd.concat([sonucUlke, sonucYas], axis=1)  
#print(s)

#s2 = pd.concat([s, sonucCinsiyet], axis=1) # cinsiyeti de ekledik
#print(s2)

# verileri egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(s, sonucCinsiyet, test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

#X_train = sc.fit_transform(x_train)
#X_test = sc.fit_transform(x_test)


