# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 16:25:47 2022

@author: reco1
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#2.veri Onisleme
#veri yukleme
veriler = pd.read_csv("maaslar.csv")
print(veriler)

#dataframe dilimleme (slice)
#egitim seviyesi x 
#maas y olarak bölüyoruz
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#Numpy array dönüsümü
X = x.values
Y = y.values

#linear regression
#dogrusal model olusturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X, Y, color="red")
plt.plot(x, lin_reg.predict(X), color="blue")

print("Linear R2 degeri")
print(r2_score(Y, lin_reg.predict(X)))

#polynomial regression
#dogrusal olmayan (nonlinear) model olusturma
#2. dereceden polinom
#herhangi bir sayıyı polinomal olarak ifade etmeye yarıyor esasında
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2) #2.dereceden obje olustur
x_poly = poly_reg.fit_transform(X) #X= 1 den 10'a kadar olan degerler 
# print(x_poly) #1 den 10'a kadar giden degerler ve onların kareleri geldi, waoww
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#4. dereceden polinom
poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 = poly_reg3.fit_transform(X)
#print(x_poly)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)


#tahminler
#linear regression tahminleri
print(lin_reg.predict([[11]])) #bir kisinin egitimseviyesi 11 ise verileceği maas 
print(lin_reg.predict([[6.6]])) #bir kisinin egitimseviyesi 6 ile 7 arasında ise verileceği maas 

#polinomal regression tahminleri
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

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

plt.figure(figsize=(8,5))
plt.scatter(x_olcekli,y_olcekli,color="red")
#her bir x_olcekli deger icin o x_olcekli degerin svr_reg de ki tahmin karsiligini bul ve ekranda goster  
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")
plt.show()

print(svr_reg.predict([[11.0]]))
print(svr_reg.predict([[6.6]]))

print("SVR R2 degeri")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


#Decision Tree Regresyonu
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4

plt.figure(figsize=(8,5))
plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")

plt.plot(x,r_dt.predict(Z),color="green")
plt.plot(x,r_dt.predict(K),color="yellow")

plt.show()
print(r_dt.predict([[11.0]]))
print(r_dt.predict([[6.6]]))
print("Decision Tree R2 degeri")
print(r2_score(Y, r_dt.predict(X))) # gercek deger ve tahmini deger

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(X,Y.ravel()) # X bilgisinden, Y bilgisini ogren

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X), color="blue")

plt.plot(X,rf_reg.predict(Z), color="green")
plt.plot(x,r_dt.predict(K),color="yellow")


print("Random Forest R2 degeri")
print(r2_score(Y, rf_reg.predict(X))) # gercek deger ve tahmini deger

print(r2_score(Y, rf_reg.predict(K))) # K icin
print(r2_score(Y, rf_reg.predict(Z))) # Z icin


#Ozet R2 degerleri
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
