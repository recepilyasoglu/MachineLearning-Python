# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:53:59 2022

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

print("Confusion")
cm = confusion_matrix(y_test, y_pred)
print(cm)


#K-NN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")#n_neighbors'a yüksek değer vermeye gerek yok, yanlış düşünce olur   
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("K-NN")
cm = confusion_matrix(y_test, y_pred)
print(cm)


#SVM
from sklearn.svm import SVC
svc = SVC(kernel="linear") #kernel olarak doğrusalı seçtik
svc.fit(X_train, y_train) #.fit her zaman train üzerine

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("SVC")
print(cm)


#NaiveBayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("GNB")
print(cm)


#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy")

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("DTC")
print(cm)


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion="entropy") 
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print("RFC")
print(cm)



