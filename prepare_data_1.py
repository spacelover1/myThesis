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
address = "/home/amin/Desktop/amin_p/pickle files/"

for pkl in os.listdir(address):

    print(pkl)
    filename = address + pkl
    pklfile = list(map(lambda x: list(x), list(pickle.load(open(filename, "rb")))[0]))
    
    my_bit_list = []
    cross_entropy_list = []
    
    for ls in pklfile:
        for hexx in ls:
            bits = bin(hexx)[2:].zfill(num_of_bits)
            num_of_one_divided_by_eight = (bits.count('1') / num_of_bits)
            my_bit_list.append(num_of_one_divided_by_eight)
     
    del pklfile    
    i = 0
    while i < len(my_bit_list)-1:
    	cross = cross_entropy([my_bit_list[i]], [my_bit_list[i + 1]])
    	cross_entropy_list.append(cross)
    	i = i + 2
        
    df = pd.DataFrame(cross_entropy_list)
    df.to_csv(filename+'.csv', mode='a', index=False, index_label=False, chunksize=1000000, header=False)
            
        
