# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:57:48 2022

@author: reco1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#veri kumesi
dataset = pd.read_csv('Social_Network_Ads.csv')
print(dataset)

X = dataset.iloc[:,[2,3]].values 
y = dataset.iloc[:,4].values 


# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=0)


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#SVM
from sklearn.svm import SVC
classifier = SVC(kernel="rbf", random_state = 0) 
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)


#ConfusionMatrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


#k-fold cross validation(k-katlamali capraz dogrulama)
from sklearn.model_selection import cross_val_score
'''
Parametreler:
1. estimator : classifier (bizim durum)
2. X
3. Y
4. cv : kaç katlamalı
'''
basari = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 4)
print(basari.mean()) #accuracy'lerin ortalaması
print(basari.std()) #ne kadar düsük cıkarsa o kadar iyidir, standart sapma

