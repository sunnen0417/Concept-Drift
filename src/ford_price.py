import numpy as np
import os
import torch.utils.data as Data

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

def get_ford_price(csv_file):
    data_tar = np.genfromtxt(csv_file, delimiter=',')
    return data_tar[:, :-1], data_tar[:, -1]

class ford_price_dataset(Data.Dataset):
    def __init__(self, time_slice = 20,
                percentile = [10, 90],
                variance_coeff = 0.5,
                normalize = True):
        super(ford_price_dataset, self).__init__()
        self.time_slice = time_slice
        self.num_batch = time_slice + 1
        self.num_class = 2
        self.basedir = 'datasets'
        self.dataset_dir = 'ford_price'
        self.normalize = normalize
        self.normalize_indices = [0, 1, 2, 3]
        self.cate_feat = [list(range(4, 41))]

        dataset_path = os.path.join(self.basedir, self.dataset_dir)
        os.system(f'mkdir -p {dataset_path}')
        os.system(f'wget -O {dataset_path}/ford_preprocessed.csv https://www.csie.ntu.edu.tw/\~b08201047/ford_preprocessed.csv')

        self.real_data, self.real_target = get_ford_price(f'{dataset_path}/ford_preprocessed.csv')
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

        # numeric: [0, 1, 2, 3]
        if t == 0:
            self.mu = np.mean(self.data[:, self.normalize_indices], axis = 0)
            self.std = np.maximum(np.std(self.data[:, self.normalize_indices]), 1e-5)

        if self.normalize:
            self.data[:, self.normalize_indices] = (self.data[:, self.normalize_indices] - self.mu) / self.std
