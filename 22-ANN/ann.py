# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:54:41 2022

@author: reco1
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
print(veriler)

#veri on isleme

X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values


#encoder: Kategorik -> Numeric
#Geography ve Gender kolonlarına label encode uyguluyoruz
from sklearn import preprocessing

#Geography
le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

#Gender
le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])


#OneHotEncode
from sklearn.preprocessing import OneHotEncoder
#ColumnsTransformer -> birden fazla kolonun aynı anda ayrı ayrı dönüştürülmesi
# ve bu features'ların(özniteliklerin) tek bir öznitelik alanında birleştirilmesini sağlıyor  
from sklearn.compose import ColumnTransformer   

#birinci kolonu veriyoruz
ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                   remainder="passthrough"
)

X = ohe.fit_transform(X)
X = X[:,1:]


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

