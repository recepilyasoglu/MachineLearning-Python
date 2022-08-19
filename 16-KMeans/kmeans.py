# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:39:37 2022

@author: reco1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")
print(veriler)

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init= "k-means++")
kmeans.fit(X)

print(kmeans.cluster_centers_)#3 tane merkez noktas verdi, n_cluster yani   


sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 123)  
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)


#1'den 10'a kadar her bir wcss değeri
plt.plot(range(1,11), sonuclar)
plt.show()


