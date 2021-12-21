import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

def one_hot(df, one_hot_features):
    for feature in one_hot_features:
        temp = pd.get_dummies(df[feature], prefix = feature)
        df = df.join(temp)
        df = df.drop(feature, axis=1)
    return df
    
class KDD99Dataset(Data.Dataset):
    def __init__(self, normalize=True):
        super(KDD99Dataset, self).__init__()
        self.base_dir = 'datasets'
        self.dataset_dir = 'kdd99'
        if os.path.exists(os.path.join(self.base_dir, self.dataset_dir)):
            print(f'{os.path.join(self.base_dir, self.dataset_dir)} exists!')
        else:
            # download the dataset
            os.makedirs(os.path.join(self.base_dir, self.dataset_dir), exist_ok=True)
            os.chdir(os.path.join(self.base_dir, self.dataset_dir))
            os.system('wget https://datahub.io/machine-learning/kddcup99/r/kddcup99.csv')  
            os.chdir('../..')
            #####
        self.alldata = []
        self.data = []
        self.target = []
        self.num_batch = 10
        self.normalize = normalize
        self.batch_data_num = 0
        self.normalize_indices = []
        self.mu = 0
        self.std = 0
        self.preprocess('datasets/kdd99/kddcup99.csv')
        self.t = 0
        self.set_t(self.t)
        self.num_class = 2
            
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return len(self.target) 

    def preprocess(self, file_path):
        df = pd.read_csv(file_path)
        one_hot_features = ['protocol_type', 'service', 'flag']
        df = one_hot(df, one_hot_features)
        f = lambda x: 0 if x == 'normal' else 1
        df['label'] = df['label'].map(f) 
        self.alldata = df.drop(['label'],axis=1).to_numpy()
        self.target = df['label'].to_numpy()
        self.batch_data_num = int(self.alldata.shape[0] / self.num_batch)

        self.normalize_indices = [0,1,2,4,5,6,7,9,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
        self.mu = np.mean(self.alldata[:self.batch_data_num, self.normalize_indices], axis = 0)
        self.std = np.maximum(np.std(self.alldata[:self.batch_data_num, self.normalize_indices]), 1e-5)

    def set_t(self, t):
        self.t = t
        start = self.t * self.batch_data_num
        end = (self.t + 1) * self.batch_data_num
        self.data = self.alldata[start:end,:]
        if self.normalize:
            self.data[:, self.normalize_indices] \
                = (self.data[:, self.normalize_indices] - self.mu) / self.std
    
