from PickleToFile import PickleToFile
from HexToBinary import HexToBinary
from math import log2
import pandas as pd


def get_value_of_list(bit_list):
    p_number = 0
    for i in bit_list:
        if i == 1:
            p_number = p_number + 1
    return p_number


def cross_entropy(p, q):
    return -sum([p[i] * log2(q[i]) for i in range(len(p))])


if __name__ == "__main__":

    file_name = 'Copy of vpn_vimeo_B.txt'
    pickle_file = 'Copy of vpn_vimeo_B.pcap.pickle'
    pk = PickleToFile(file_name, pickle_file)
    pk.create_pickle_file()
    h = HexToBinary(file_name)
    hex_list = h.read_file()
    num_of_bits = 8

    scale = 16
    bin_data = []
    for i in hex_list:
        bin_data.append(bin(int(i, scale))[2:].zfill(num_of_bits))

    my_bit_list = []
    for byte in bin_data:
        bit_list = []
        for bit in byte:
            bit_list.append(int(bit))
        num_of_one_divided_by_eight = get_value_of_list(bit_list) / 8
        my_bit_list.append(num_of_one_divided_by_eight)

    cross_entropy_list = []
    i = 0
    while i < len(my_bit_list):
        cross = cross_entropy([my_bit_list[i]], [my_bit_list[i + 1]])
        cross_entropy_list.append(cross)
        i = i + 2

    df = pd.DataFrame(cross_entropy_list)
    df.to_csv(r'Copy of vpn_vimeo_B.csv', index=False, index_label=False, chunksize=1000000, header=False)
