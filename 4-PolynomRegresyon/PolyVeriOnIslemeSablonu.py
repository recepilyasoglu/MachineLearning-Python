# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 02:56:00 2022

@author: reco1
"""
#1.kütüphaneler
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

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

# Gorsellestirme
# plt.scatter(X,Y,color="red")
# plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color="blue")
# plt.show()

# plt.scatter(X,Y, color="red")
# plt.plot(x, lin_reg.predict(X), color="blue") # herbir x'e karsilik gelen tahminleri gorsellestirme
# plt.show()

plt.scatter(X,Y,color="red")
plt.plot(X, lin_reg3.predict(poly_reg3.fit_transform(X)), color="blue")
plt.show()


#tahminler
#linear regression tahminleri
print(lin_reg.predict([[11]])) #bir kisinin egitimseviyesi 11 ise verileceği maas 
print(lin_reg.predict([[6.6]])) #bir kisinin egitimseviyesi 6 ile 7 arasında ise verileceği maas 

#polinomal regression tahminleri
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))



