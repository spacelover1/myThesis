import pandas as pd
from math import log2
from PickleToFile import PickleToFile
import time

def cross_entropy(p, q):
	return -sum([p[i] * log2(q[i]) for i in range(len(p))])

start = time.time()
name_pickle = 'Copy of vpn_vimeo_B.pcap.pickle'
name_txt = name_pickle.replace('.pickle', '.text')
pk = PickleToFile(name_txt, name_pickle)
pk.create_pickle_file()

num_of_bits = 8
scale = 16
my_bit_list = []
cross_entropy_list = []
name_csv = name_pickle.replace('.pickle', '.csv')

with open(name_txt, 'r') as f:
	for line in f:
		for char in line:
			bits = bin(ord(char))[2:].zfill(num_of_bits)
			num_of_one_divided_by_eight = bits.count('1') / 8
			my_bit_list.append(num_of_one_divided_by_eight)
			
		if len(my_bit_list) > 10000000 and len(my_bit_list)%2 == 0:
			cross_entropy_list = []
			i = 0
			while i+1 < len(my_bit_list):
				cross = cross_entropy([my_bit_list[i]], [my_bit_list[i + 1]])
				cross_entropy_list.append(cross)
				i = i + 2
			df = pd.DataFrame(cross_entropy_list)
			df.to_csv(name_csv, mode='a', index=False, index_label=False, chunksize=1000000, header=False)
			del my_bit_list
			my_bit_list = []
			del cross_entropy_list
	
	if len(my_bit_list) > 1:
		cross_entropy_list = []
		i = 0
		while i+1 < len(my_bit_list):
			cross = cross_entropy([my_bit_list[i]], [my_bit_list[i + 1]])
			cross_entropy_list.append(cross)
			i = i + 2
		df = pd.DataFrame(cross_entropy_list)
		df.to_csv(name_csv, mode='a', index=False, index_label=False, chunksize=1000000, header=False)

print(time.time() - start)