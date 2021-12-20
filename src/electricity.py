import numpy as np
import pandas as pd
import os
import torch
import torch.utils.data as Data

class ElectricityDataset(Data.Dataset):

    def __init__(self, normalize = True):
        super(ElectricityDataset, self).__init__()
        self.base_dir = 'datasets'
        self.dataset_dir = 'electricity'

        check_create_dir(self.base_dir)
        os.chdir(self.base_dir)
        check_create_dir(self.dataset_dir)
        os.chdir(self.dataset_dir)
        if not os.path.exists('electricity.csv'):
            os.system('wget https://datahub.io/machine-learning/electricity/r/electricity.csv')

        # self.num_batch = 943

        self.all_data, self.data = [], []
        self.all_target, self.target = [], []
        self.generate('electricity.csv')

        os.chdir('../../')

        self.normalize = normalize
        self.t = 0
        self.set_t(self.t)
        self.num_class = 2

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return self.target.shape[0]

    def generate(self, file_path):
        df = pd.read_csv(file_path)
        df['class'] = pd.factorize(df['class'])[0]
        df = one_hot(df, ['day'])

        # put 'class' column to the last column
        df_class = df.reindex(columns=['class'])
        df = df.drop(columns = ['class'])
        df = df.merge(df_class, left_index = True, right_index = True)

        data = df.to_numpy()

        _ = 0
        for i in range(1, data.shape[0]):
            if data[i - 1][0] != data[i][0]:
                self.all_data.append(data[_:i, 1:-1])
                self.all_target.append(data[_:i, -1])
                _ = i + 1

        self.all_data.append(data[_:, 1:-1])
        self.all_target.append(data[_:, -1])

        self.num_batch = len(self.all_data)

        # numeric features: [0, 1, 2, 3, 4, 5]
        self.normalize_indices = [0, 1, 2, 3, 4, 5]
        self.mu = np.mean(self.all_data[0][:, self.normalize_indices], axis = 0)
        self.std = np.maximum(np.std(self.all_data[0][:, self.normalize_indices]), 1e-5)

        return

    def set_t(self, t):
        self.t = t
        self.data = np.array(self.all_data[t], dtype='float32')
        self.target = np.array(self.all_target[t], dtype='int64')

        if self.normalize:
            self.data[:, self.normalize_indices] \
                = (self.data[:, self.normalize_indices] - self.mu) / self.std

        return
