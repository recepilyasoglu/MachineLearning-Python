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



