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
    
    pklfile_bin = []
    for ls in pklfile:
        for byte in ls:
            pklfile_bin.append(bin(byte)[2:].zfill(num_of_bits)) 
     
    del pklfile  
     
    
    cross_entropy_list_1 = []
    cross_entropy_list_2 = []
    
    for i in range(1, np.shape(pklfile_bin)[0]-1):
        #if i < limitation:
        
            #print(str(i+1) + '/' + str(np.shape(pklfile_bin)[0]-2))
           
        cross_entropy_list_1.append(cross_entropy([(pklfile_bin[i-1][-overlap:] + pklfile_bin[i][0:overlap]).count('1') / num_of_bits],
                                                  [pklfile_bin[i].count('1') / num_of_bits]))
        cross_entropy_list_2.append(cross_entropy([pklfile_bin[i].count('1') / num_of_bits],
                                                  [(pklfile_bin[i][-overlap:] + pklfile_bin[i+1][0:overlap]).count('1') / num_of_bits]))
    
    del pklfile_bin
    
    if overlap == 0:    
        df = pd.DataFrame(cross_entropy_list_1)
        df.to_csv(filename+'.csv', mode='a', index=False, index_label=False, chunksize=1000000, header=False)
        
    else:
        df = pd.DataFrame(np.concatenate(
                          (np.expand_dims(np.array(cross_entropy_list_1), axis=1),
                           np.expand_dims(np.array(cross_entropy_list_2), axis=1)),
                           axis=1))
        
        df.to_csv(filename+'.csv', mode='a', index=False, index_label=False, chunksize=1000000, header=False)

        
        
