# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:57:11 2022

@author: reco1
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#2.veri onisleme
#2.1.veri yukleme
data = pd.read_csv("satislar.csv")
# data = pd.read_csv("veriler.csv")
print(data)

# aylar bagimsiz degisken
aylar = data[["Aylar"]]
print(aylar)

satislar = data[["Satislar"]]
print(satislar)


# verileri egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# olay -> aylara gore satis tahmini 

# SimpleLinearRegression
# model insası(linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)














