# My Masters Thesis

## Title: Encrypted Network Traffic Classification


Using **cross-entropy on network traffic** is the novelty of my work. 

Dataset: [VPN-nonVPN dataset (ISCXVPN2016)](https://www.unb.ca/cic/datasets/vpn.html)

I have used the mentioned dataset for my thesis research. This dataset contains pcap files, I have converted these files to pickle files and then read the numbers in files, which represents the traffic data for each app, then I have calculated cross-entropy of those numbers using sliding windows with different window size; four window size have been used here: 2, 4, 6, 8.

### Base paper
The best deep learning model that could give the most feasible answer is proposed in [this paper](https://arxiv.org/pdf/1709.02656.pdf).

And [here](https://github.com/spacelover1/deeppacket) is the implementation of the Deep Learning model.


### Future work:

Using other window sizes.
Using sliding window on some part of traffic instead of all of it.

