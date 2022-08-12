# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 02:21:05 2022

@author: reco1
"""

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

# plt.scatter(X,Y,color="red")
# plt.plot(X, lin_reg3.predict(poly_reg3.fit_transform(X)), color="blue")
# plt.show()


#tahminler
#linear regression tahminleri
print(lin_reg.predict([[11]])) #bir kisinin egitimseviyesi 11 ise verileceği maas 
print(lin_reg.predict([[6.6]])) #bir kisinin egitimseviyesi 6 ile 7 arasında ise verileceği maas 

#polinomal regression tahminleri
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


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

plt.scatter(x_olcekli,y_olcekli,color="red")
#her bir x_olcekli deger icin o x_olcekli degerin svr_reg de ki tahmin karsiligini bul ve ekranda goster  
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")

print(svr_reg.predict([[11.0]]))
print(svr_reg.predict([[6.6]]))


#benim poly üzerinde denemem
sc3 = StandardScaler()
x_olcekli2 = sc3.fit_transform(X)
sc4 = StandardScaler()
y_olcekli2 = sc4.fit_transform(Y)
                              
svr_reg2 = SVR(kernel = "poly")
svr_reg2.fit(x_olcekli2, y_olcekli2) 

plt.scatter(x_olcekli2,y_olcekli2,color="red")
#her bir x_olcekli deger icin o x_olcekli degerin svr_reg de ki tahmin karsiligini bul ve ekranda goster  
plt.plot(x_olcekli2,svr_reg2.predict(x_olcekli2),color="blue")

print(svr_reg2.predict([[15.0]]))
print(svr_reg2.predict([[4.6]]))

