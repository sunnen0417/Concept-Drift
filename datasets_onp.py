import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

class ONPDataset(Data.Dataset):
    def __init__(self, normalize=False):
        super(ONPDataset, self).__init__()
        self.base_dir = 'datasets'
        self.dataset_dir = 'onp'
        if os.path.exists(os.path.join(self.base_dir, self.dataset_dir)):
            print(f'{os.path.join(self.base_dir, self.dataset_dir)} exists!')
        else:
            # download the dataset
            os.makedirs(self.base_dir, exist_ok=True)
            os.chdir(self.base_dir)
            os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip')
            os.system('unzip OnlineNewsPopularity.zip')
            os.system('mv OnlineNewsPopularity onp')
            os.system('rm -f OnlineNewsPopularity.zip')
            os.chdir('..')
            
        self.df = pd.read_csv(os.path.join(self.base_dir, self.dataset_dir, 'OnlineNewsPopularity.csv'))
        self.df['popular'] = [1 if sh >= 1400 else 0 for sh in self.df[' shares']]
        self.month_list = self.df.url.str.slice(20, 27).unique()
        self.normalize_indices = list(range(0,11)) +list(range(17,29)) + list(range(37,58)) 
            
        self.num_batch = len(self.month_list)
        self.data = []
        self.target = []
        self.normalize= normalize
        self.mu = None
        self.std = None
        self.t = 0
        self.set_t(self.t)
        self.num_class = 2
            
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return len(self.target) 
    
    def set_t(self, t):
        self.t = t
        self.data = []
        self.target = []
        
        keep = self.df.url.str.slice(20, 27) == self.month_list[self.t]
        self.target = self.df.values[keep, -1]
        self.data = self.df.values[keep, 2:-2]         
        
        self.data = np.array(self.data, dtype='float32')
        self.target = np.array(self.target, dtype='int64')
        
        if t == 0:
            self.mu = np.mean(self.data[:, self.normalize_indices], axis=0)
            self.std = np.maximum(np.std(self.data[:, self.normalize_indices], axis=0), 1e-5)
            
        if self.normalize:
            self.data[:, self.normalize_indices] = (self.data[:, self.normalize_indices] - self.mu) / self.std
