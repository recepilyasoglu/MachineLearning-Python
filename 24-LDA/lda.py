# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:54:23 2022

@author: reco1
"""

#kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#veri kumesi

veriler = pd.read_csv('Wine.csv')
print(veriler)

X = veriler.iloc[:,0:13].values 
y = veriler.iloc[:,13].values 

#eğitim ve test kümlerinin bölünmesi
from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)#boyut sayısı, kaç boyuta indirgenmesini istiyorsak

X_train2 = pca.fit_transform(X_train)#2 boyuta indirgendi
X_test2 = pca.transform(X_test)

#pca dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0) #♣ her logistic regresyon kullandığımızda verdiğimiz sabit değeri kullan   
classifier.fit(X_train, y_train)

#pca dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

#tahminler
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
#actual / PCA olmadan çıkan sonuç
print("Gerçek / PCA'siz")
cm = confusion_matrix(y_test, y_pred)
print(cm)

#actual / PCA sonrası çıkan sonuç
print("Gerçek / PCA ile")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

#PCA sonrası / PCA öncesi
print("PCA'siz / PCA ile")
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)


#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)

#lda çalışabilmesi sınıfları öğrenmesi gerkiyor,
#onun için pca'den farklı olarak iki parametre verdik(X_train,y_train)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

#LDA donusumunden sonra
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda, y_train)

#LDA verisini tahmin et
y_pred_lda = classifier_lda.predict(X_test_lda)

#LDA sonrası / orijinal
print("LDA / Orjinal ile")
cm4 = confusion_matrix(y_pred, y_pred_lda)
print(cm4)

#LDA -> sistemimiz %100 başarı ile çalıştı
#PCA -> uyguladğımızda sınıfları gözetmeden bunları ayrıştırdığı için hata yapma ihtimali daha yüksek 
#PCA -> boyut indirgemede avantaj sağlıyor ama başarıya negatif etki ediyor
