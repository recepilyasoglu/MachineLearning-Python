# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 01:45:05 2022

@author: reco1
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv("odev_tenis.csv")
# data = pd.read_csv("veriler.csv")
print(veriler)


#encoder: Kategorik -> Numeric 
#bütün kolonlar üzerine LabelEncoder.fit_transform'u etmiş, kolay yolu
from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1] # ilk kolonu aldık(outlook)

from sklearn import preprocessing
#OneHotEncoding
ohe = preprocessing.OneHotEncoder()
# bi onceki asamada sayıya ulke kolonumuz ogrenilip bunu transform edecek
# 3 kolnu sırasıyla 0 ve 1 lerden oluscak sekilde OneHotEncode'a donusturecek
# toarray le numpy array olarak sonucu alıyoruz
c = ohe.fit_transform(c).toarray()
print(c)


havadurumu = pd.DataFrame(data=c, index=range(14), columns=["overcast","rainy","sunny"])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1) #outlook kolonunu
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1) #windy,play kolonunu da ekledik 


#humidity'i tahmin etmek ? 

# verileri egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

#son kolona kadar olanlar bagımsız degisken
#son kolon ise bagımlı degisken
x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1], sonveriler.iloc[:,-1:], test_size=0.33, random_state=0)   


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#x'in test olarak ayrılmış kısmını, yukardakine göre predict et, ve y_predict'e yaz 
y_pred = regressor.predict(x_test)
print(y_pred) # ilk prediction değerleri

#backward elimination
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)

X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values # bagımsız degiskenleri, son kolona kadar al
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit() # bagımlı degiskeni(son kolonu) al 
print(model.summary())

# P value çok yüksek geldi onun için, windy kolonunu(x1) atıcaz
# 1.sütundan son sütuna kadar olanları al
sonveriler = sonveriler.iloc[:,1:] # windy kolonunu attık

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)

X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())


x_train = x_train.iloc[:,1:] # x_train içinden windy gitti, tekrar model oluşturmak için
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train) #burdaki train değeri ile sistemi tekrar eğit
y_pred = regressor.predict(x_test) # ve yeni test değeri ile prediction oluştur


