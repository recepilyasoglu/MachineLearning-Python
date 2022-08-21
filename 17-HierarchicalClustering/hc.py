# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 21:21:21 2022

@author: reco1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")
print(veriler)

X = veriler.iloc[:,3:].values

#kmeans
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init= "k-means++")
kmeans.fit(X)

print(kmeans.cluster_centers_)#3 tane merkez noktas verdi, n_cluster yani   


sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 123)  
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    
plt.plot(range(1,11), sonuclar)
plt.show()

kmeans = KMeans(n_clusters = 4, init = "k-means++", random_state = 123)  
Y_tahmin = kmeans.fit_predict(X)
print(Y_tahmin)

plt.figure(figsize=(8,5))
plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1], s=100, c="red")
plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1], s=100, c="blue")
plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1], s=100, c="green")
plt.scatter(X[Y_tahmin==3,0], X[Y_tahmin==3,1], s=100, c="yellow")

plt.title("KMeans")
plt.show()


#HC(HierarchicalClustering)
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 4, affinity = "euclidean", linkage = "ward" )      
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)

plt.figure(figsize=(8,5))
plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1], s=100, c="red")
plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1], s=100, c="blue")
plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1], s=100, c="green")
plt.scatter(X[Y_tahmin==3,0], X[Y_tahmin==3,1], s=100, c="yellow")

plt.title("HC")
plt.show()


#with scipy
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(8,5))
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.show()
