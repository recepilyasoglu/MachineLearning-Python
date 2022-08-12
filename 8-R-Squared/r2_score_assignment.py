# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:07:57 2022

@author: reco1
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

#veri yukleme
veriler = pd.read_csv("maaslar_yeni.csv")
print(veriler)

#nan degerin temizlenmesi
veriler.drop([12], axis=0, inplace=True)

# bagımsız degiskenler = UnvanSeviyesi,Kidem,Puan
# bagımlı(bulunmak istenen) degisken = maas 

x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values

#bagimsiz degiskenler arasindaki iliskiyi verir
print(veriler.corr())

#linear regression
#dogrusal model olusturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

print("Linear OLS")
model = sm.OLS(lin_reg.predict(X),X)#X prediction sonuclarini al X le karsilastir 
print(model.fit().summary())

print("Linear R2 degeri")
print(r2_score(Y, lin_reg.predict(X)))


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
#4. dereceden polinom
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
#print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


#tahminler


print("Poly OLS")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

print("Polynomial R2 degeri")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)


from sklearn.svm import SVR
#Amac;iki deger arasındaki iliskiyi kurması, bulması
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_olcekli, y_olcekli) 


print("SVR OLS")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())

print("SVR R2 degeri")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


#Decision Tree Regresyonu
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)


print("DecisionTree OLS")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print("Decision Tree R2 degeri")
print(r2_score(Y, r_dt.predict(X)))


#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(X,Y.ravel()) # X bilgisinden, Y bilgisini ogren


print("Random Forest OLS")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())


print("Random Forest R2 degeri")
print(r2_score(Y, rf_reg.predict(X))) # gercek deger ve tahmini deger


print("----------------------")
print("Linear R2 degeri")
print(r2_score(Y, lin_reg.predict(X)))

print("Polynomial R2 degeri")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print("SVR R2 degeri")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

print("Decision Tree R2 degeri")
print(r2_score(Y, r_dt.predict(X)))

print("Random Forest R2 degeri")
print(r2_score(Y, rf_reg.predict(X)))


