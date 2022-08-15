# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 15:56:48 2022

@author: reco1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)


x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x, y, test_size=0.33, random_state=0)             

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#LogisticRegression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0) 
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test) # kıyaslama yapabilmek adına


#ConfusionMatrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


#K-NN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")#n_neighbors yüksek değer vermeye gerek yok yanlış düşünce olur   
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

