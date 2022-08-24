# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:06:38 2022

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

#Thompson
import random 

# UCB
N = 10000  # 10.000 tıklama
d = 10  # toplam 10 ilan var
# Ni(n)
toplam = 0  # toplam odul
secilenler = []
birler = [0] * d
sifirlar = [0] * d

for n in range(1, N):
    ad = 0  # secilen ilan
    max_th = 0
    for i in range(0, d):
        rasbeta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n, ad]  # verilerdeki n.satır = 1 ise odul 1
    if odul == 1:
        birler[ad] = birler[ad] + 1
    else:
        sifirlar[ad] = sifirlar[ad] + 1
        
    toplam = toplam + odul

print("Toplam Odul:")
print(toplam)

plt.figure(figsize=(8, 5))
plt.hist(secilenler)
plt.show()


