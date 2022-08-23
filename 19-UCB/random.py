# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:50:06 2022

@author: reco1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

import random 

N = 10000
d = 10
toplam = 0
secilenler = []

for n in range(0,N):
    ad = random.randrange(d) # 10 tane ilan var, 10'a kadar bir değer üret, tıklanan ilan olacak bu      
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeli n. satır = 1 ise odul 1
    toplam = toplam + odul # toplam odulu verecek, her seferinde odul olarak tıklanan deger bize odul olarak gelecek     


# her bir ilandan ne kadar secildiği
plt.figure(figsize=(8,5))
plt.hist(secilenler)
plt.show()


