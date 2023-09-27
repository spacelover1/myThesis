# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 02:13:48 2020

@author: AMIN
"""
import numpy as np
import pandas as pd
from math import log2
import time
import pickle
import os

def cross_entropy(p, q):
    
    for i in range(len(q)):
        if q[i] == 0:
            q[i] += 1
            p[i] = 0
            
    return round(-sum([p[i] * log2(q[i]) for i in range(len(p))]), 2)


num_of_bits = 8
overlap = 0
#limitation = 20000000
address = "/home/amin/Desktop/amin_p/pickle files/"

for pkl in os.listdir(address):

    print(pkl)
    filename = address + pkl
    pklfile = list(map(lambda x: list(x), list(pickle.load(open(filename, "rb")))[0]))
    
    pklfile_2 = []
    for ls in pklfile:
        for byte in ls:
            pklfile_2.append(byte) 
     
    del pklfile  
    pklfile_2 = list(map(lambda x: x/255, pklfile_2))
    
    df = pd.DataFrame(pklfile_2)
    df.to_csv(filename+'.csv', mode='a', index=False, index_label=False, chunksize=1000000, header=False)
    del pklfile_2  
  
    
    
        
        
