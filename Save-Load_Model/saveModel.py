# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 16:04:08 2022

@author: reco1
"""

import pandas as pd


url = "http://bilkav.com/satislar.csv"

veriler = pd.read_csv(url)
veriler = veriler.values

X = veriler[:,0:1]
Y = veriler[:,1]

bolme = 0.33

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = bolme)


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, Y_train)
print(lr.predict(X_test))

import pickle 

dosya = "model.kayit"

#save
pickle.dump(lr, open(dosya,"wb"))#write-binary

#load
yuklenen = pickle.load(open(dosya,"rb")) #read-binary
print("Yuklenen")
print(yuklenen.predict(X_test)) #dışardan okunan modeli tahmin et

