# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:59:59 2022

@author: reco1
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

'''
# Random Selection (Rastgele Seçim)

import random 

N = 10000
d = 10
toplam = 0
secilenler = []

for n in range(0,N):
    ad = random.randrange(d) # 10 tane ilan var, 10'a kadar bir değer üret, tıklanan ilan olacak bu      
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    toplam = toplam + odul # toplam odulu verecek, her seferinde odul olarak tıklanan deger bize odul olarak gelecek     


# her bir ilandan ne kadar secildiği
plt.figure(figsize=(8,5))
plt.hist(secilenler)
plt.show()

'''

# UCB
N = 10000  # 10.000 tıklama
d = 10  # toplam 10 ilan var
# Ri(n)
# 10 elemanlı her elemanı 0 olan dizi, herhangi bir odul yok baslangıcta
oduller = [0] * d
# Ni(n)
tiklamalar = [0] * d  # o ana kadarki tiklamalar
toplam = 0  # toplam odul
secilenler = []

for n in range(1, N):
    ad = 0  # secilen ilan
    max_ucb = 0
    for i in range(0, d):  # butun ilanların teker teker ihtimallerine bakıyoruz
        if(tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2 * math.log(n) / tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N * 10
        if max_ucb < ucb:  # max'tan büyük bir ucb çıktı
            max_ucb = ucb
            ad = i          
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n, ad]  # verilerdeki n.satır = 1 ise odul 1
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul
    
print("Toplam Odul:")
print(toplam)

plt.figure(figsize=(8,5))
plt.hist(secilenler)
plt.show()

