import torch
import pandas as pd
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt

def get_ford_price(csv_file):
    data_tar = np.genfromtxt(csv_file, delimiter=',')
    return data_tar[:, :-1], data_tar[:, -1]

def get_threshold(target_range, variance_coeff, slices):
    thresholds = []
    u = (target_range[1] - target_range[0]) / slices
    variant_step = np.random.normal(u, u * variance_coeff, slices + 1)
    variant_step[0] = 0 # no need to move at starting threshold
    t = target_range[0]
    for i in range(slices + 1):
        t += variant_step[i]
        thresholds.append(t)
    return thresholds

class ford_price_dataset(Data.Dataset):
    def __init__(self, time_slice = 20,
                csv_file = '../data/ford_preprocessed.csv',
                percentile = [10, 90],
                variance_coeff = 0):
        super(ford_price_dataset, self).__init__()
        self.time_slice = time_slice

        self.real_data, self.real_target = get_ford_price(csv_file)
        self.data, self.target = [], []
        self.dim = self.real_data.shape[1]
        self.target_range = [
            np.percentile(np.sort(self.real_target), percentile[0]),
            np.percentile(np.sort(self.real_target), percentile[1])
        ]

        self.thresholds = get_threshold(self.target_range, variance_coeff, self.time_slice)

        self.n_sample = self.real_target.shape[0]
        self.t = 0
        self.set_t(self.t)

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return int(self.n_sample * 0.7)

    def set_t(self, t):
        indices = np.random.choice(range(self.n_sample), int(self.n_sample * 0.7), replace=False)

        self.t = t
        data = self.real_data[indices]
        target = []

        for i in list(indices):
            target.append(self.real_target[i] >= self.thresholds[t])

        self.data = np.array(data)
        self.target = np.array(target, dtype = "int64")

def info(trainset, time_slice):
    cnt = []
    batch_size = []
    for t in range(time_slice):
        trainset.set_t(t)
        c0 = (trainset.target == 0).sum()
        c1 = (trainset.target == 1).sum()
        cnt.append(c1 - c0)
        batch_size.append(trainset.target.shape[0])
    print("#label1 - #label0:", cnt)
    print("trainset size:", batch_size)
    print("target range:", trainset.target_range)

def draw_data(trainset):
    value = trainset.real_target    
    target_range = [
            np.percentile(np.sort(value), 10),
            np.percentile(np.sort(value), 90)
        ]   
    bins = np.arange(min(value), target_range[0], 200)
    plt.hist(value, bins, color = 'yellowgreen')
    bins = np.arange(target_range[0], target_range[1], 200)
    plt.hist(value, bins, color = 'mediumslateblue')
    bins = np.arange(target_range[1], max(value) + 500, 200)
    plt.hist(value, bins, color = 'lightcoral')
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.savefig('ford_data.png')

if __name__ == '__main__':
    time_slice = 20
    trainset = ford_price_dataset(time_slice, variance_coeff = 0.5)
    info(trainset, time_slice)
    trainset = ford_price_dataset(time_slice, variance_coeff = 0)
    info(trainset, time_slice)
    draw_data(trainset)
