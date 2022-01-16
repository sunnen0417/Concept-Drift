import numpy as np
import pandas as pd
import copy
import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
        
class SoftmaxDataset(Data.Dataset):
    def __init__(self, softmax_data, mode='train'):
        super(SoftmaxDataset, self).__init__()
        self.softmax_data = softmax_data
        self.mode = mode
        
    def __getitem__(self, index):
        if self.mode == 'train':
            data = [self.softmax_data[i][index] for i in range(len(self.softmax_data)-1)]
            data = torch.FloatTensor(data)
            target = self.softmax_data[-1][index]
            target = torch.FloatTensor(target)
            return data, target
        else:
            data = [self.softmax_data[i][index] for i in range(len(self.softmax_data))]
            data = torch.FloatTensor(data)
            return data
        
    def __len__(self):
        return len(self.softmax_data[0])    

class BufferDataset(Data.Dataset):
    def __init__(self, X, y, target_type='hard'):
        super(BufferDataset, self).__init__()
        self.data = torch.FloatTensor(X)
        self.target_type = target_type
        if self.target_type == 'hard':
            self.target = np.array(y, dtype='int64')
        else:
            self.target = torch.FloatTensor(y)
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.target) 

class StoreDataset(Data.Dataset):
    def __init__(self):
        super(StoreDataset, self).__init__()
        self.data = None
        self.target = None

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return len(self.target)

    def append(self, new_data, new_target):
        if self.data is None:
            self.data = new_data
            self.target = new_target
        else:
            self.data = np.r_[self.data, new_data]
            self.target = np.r_[self.target, new_target]

def get_translate_concept(s, e, slice):
    concept = []
    for i in range(slice+1):
        t = s[2] + i * (e[2] - s[2]) / slice
        concept.append([s[0], s[1], t])
    return concept

def get_rotate_concept(angles):
    concept = []
    for theta in angles:
        concept.append([math.sin(theta), -math.cos(theta), 0])
    return concept

def get_uncertain_translate_concept(s, e, slice, std_ratio=0.5):
    concept = []
    u = (e[2] - s[2]) / slice
    steps = np.random.normal(u, u*std_ratio, slice+1)
    steps[0] = 0    
    t = s[2]
    for i in range(slice+1):
        t += steps[i]
        concept.append([s[0], s[1], t])
    return concept

def get_hyperball_concept(r_range=None, c_range=None, K_range=None, t=11):
    """
    Args:
        r_range: the range of each dimension of radius
            * size -> 2 * dimension
        c_range: the range of each dimension of center
            * size -> 2 * dimension
        K_range:
            * size -> dimension
        t:
            number of time
    """

    if r_range == None:
        # default generate
        assert c_range == None
        assert K_range == None
        radius = list(np.linspace([2, 1], [1, 2], t))
        center = list(np.linspace([0, -15], [0, 15], t))
        K = [25] * t
    else:
        radius = list(np.linspace(r_range[0], r_range[1], t))
        center = list(np.linspace(c_range[0], c_range[1], t))
        K = list(np.linspace(K_range[0], K_range[1], t))

    return [radius, center, K]

# Translation dataset (Synthetic): -x+4y-20 -> -x+4y+20
class TranslateDataset(Data.Dataset):
    def __init__(self):
        super(TranslateDataset, self).__init__()
        self.x_range = [-20, 20]
        self.y_range = [-5, 5]
        s = [-1, 4, -20]
        e = [-1, 4, 20]
        self.num_batch = 11
        concept = get_uncertain_translate_concept(s, e, self.num_batch-1, std_ratio=0.5) 
        self.n_per_batch = 4000
        self.concept = np.array(concept)
        self.t = 0
        self.data = []
        self.target = []
        self.set_t(self.t)
        self.num_class = 2
        self.cate_feat = []
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return len(self.target)
    
    def set_t(self, t):
        self.t = t
        self.data = np.random.uniform([self.x_range[0], self.y_range[0]], [self.x_range[1], self.y_range[1]],
                                 size=(self.n_per_batch, 2))
        w = self.concept[t].reshape((-1, 1))
        aug_data = np.concatenate((self.data, np.ones((len(self.data), 1))), axis=1)
        y = np.matmul(aug_data, w).reshape((-1,))
        dist = np.absolute(y) / np.linalg.norm(self.concept[t][0:2])
        y = y >= 0
        y = np.array(y, dtype='int64')
        u = np.mean(dist)
        std = np.std(dist)
        related_dist = (dist - u) / std
        for i in range(len(related_dist)):
            if related_dist[i] < -0.75:
                p = np.random.uniform(0, 1)
                if p < 0.1:
                    if y[i] == 0:
                        y[i] = 1
                    else:
                        y[i] = 0
        self.target = y

# Rotation dataset (Synthetic): rotate 2 circles
class RotateDataset(Data.Dataset):
    def __init__(self):
        super(RotateDataset, self).__init__()
        self.x_range = [-5, 5]
        self.y_range = [-5, 5]
        self.num_batch = 41
        rpc = 4 * math.pi / (self.num_batch - 1)
        angles = []
        for i in range(self.num_batch):
            angles.append(i*rpc)
        concept = get_rotate_concept(angles)
        self.num_batch = 41
        self.n_per_batch = 4000
        self.concept = np.array(concept)
        self.t = 0
        self.data = []
        self.target = []
        self.set_t(self.t)
        self.num_class = 2
        self.cate_feat = []

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return len(self.target)
    
    def set_t(self, t):
        self.t = t
        self.data = np.random.uniform([self.x_range[0], self.y_range[0]], [self.x_range[1], self.y_range[1]],
                                 size=(self.n_per_batch, 2))
        w = self.concept[t].reshape((-1, 1))
        aug_data = np.concatenate((self.data, np.ones((len(self.data), 1))), axis=1)
        y = np.matmul(aug_data, w).reshape((-1,))
        dist = np.absolute(y) / np.linalg.norm(self.concept[t][0:2])
        y = y >= 0
        y = np.array(y, dtype='int64')
        u = np.mean(dist)
        std = np.std(dist)
        related_dist = (dist - u) / std
        for i in range(len(related_dist)):
            if related_dist[i] < -0.75:
                p = np.random.uniform(0, 1)
                if p < 0.1:
                    if y[i] == 0:
                        y[i] = 1
                    else:
                        y[i] = 0
        self.target = y

# Hyperball dataset (Synthetic): x1^2/20^2+(x2-20)^2/15^2=1→x1^2/2^2+(x2-20)^2/30^2=1
class HyperballDataset(Data.Dataset):
    """
    Args:
        ranges:
            * range of each dimension
            * size -> 2 (low and high) * dimension
        n_per_batch:
            * number of sample per batch
        concept:
            * [r, c, K]
            * r -> raidus of each dimension
                * size: time * dimension
            * c -> center of each dimension
                * size: time * dimension
            * K -> value for easier adjustment
                * size: time
        noise:
            * probability that label is flipped
              while generating data
    """
    """
    decision boundary of hyper ball:
        * sum((x-c)**2 /r**2) = K
    """

    def __init__(self):
        super(HyperballDataset, self).__init__()
        self.ranges = [[-10,-10],[10,10]]
        self.x_range = [self.ranges[0][0], self.ranges[1][0]]
        self.y_range = [self.ranges[0][1], self.ranges[1][1]]
        self.dim = len(self.ranges[0])
        self.num_batch = 11
        self.n_per_batch = 4000
        self.concept = get_hyperball_concept(r_range=[[20,15],[2,30]], c_range=[[0,20],[0,20]], K_range=[1, 1], t=self.num_batch)
        self.radius = self.concept[0]
        self.center = self.concept[1]
        self.K = self.concept[2]
        self.noise = 0
        self.t = 0
        self.data = []
        self.target = []
        self.set_t(self.t)
        self.num_class = 2
        self.cate_feat = []
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return len(self.target)

    def set_t(self, t):
        self.t = t
        self.data = np.random.uniform(self.ranges[0], self.ranges[1],
                                 size=(self.n_per_batch, self.dim))
        radius = np.array(self.radius[t])
        center = np.array(self.center[t])
        K = self.K[t]
        y = np.sum((self.data - center)**2 / radius**2, axis = 1) - K
        y = y * np.random.uniform(-self.noise, 1 - self.noise, self.n_per_batch)
        y = y >= 0
        y = np.array(y, dtype='int64')
        self.target = y

# Gas sensor dataset
class GasSensorDataset(Data.Dataset):
    def __init__(self, normalize=True):
        super(GasSensorDataset, self).__init__()
        self.base_dir = 'datasets'
        self.dataset_dir = 'gas'
        if os.path.exists(os.path.join(self.base_dir, self.dataset_dir)):
            print(f'{os.path.join(self.base_dir, self.dataset_dir)} exists!')
        else:
            # download the dataset
            os.makedirs(self.base_dir, exist_ok=True)
            os.chdir(self.base_dir)
            os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip')
            os.system('unzip Dataset.zip')
            os.system('mv Dataset gas')
            os.system('rm -f Dataset.zip')
            os.chdir('..')
            
        self.num_batch = 10
        self.data = []
        self.target = []
        self.normalize= normalize
        self.mu = None
        self.std = None
        self.t = 0
        self.set_t(self.t)
        self.num_class = 6
        self.cate_feat = []
            
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return len(self.target) 
    
    def set_t(self, t):
        self.t = t
        self.data = []
        self.target = []
        with open(os.path.join(self.base_dir, self.dataset_dir, f'batch{t+1}.dat'), 'r') as f:
            while 1:
                line = f.readline()
                # EOF
                if line == '':
                    break
                a = line.split()
                self.target.append(int(a[0])-1)
                l = []
                for i in range(1, len(a)):
                    j = a[i].find(':')
                    l.append(float(a[i][j+1:]))
                self.data.append(l)
        self.data = np.array(self.data, dtype='float32')
        self.target = np.array(self.target, dtype='int64')
        
        if t == 0:
            self.mu = np.mean(self.data, axis=0)
            self.std = np.maximum(np.std(self.data, axis=0), 1e-5)
            
        if self.normalize:
            self.data = (self.data - self.mu) / self.std


# covertype dataset
class CovertypeDataset(Data.Dataset):
    def __init__(self, normalize=True):
        super(CovertypeDataset, self).__init__()
        self.base_dir = 'datasets'
        self.dataset_dir = 'covertype'
        if os.path.exists(os.path.join(self.base_dir, self.dataset_dir)):
            print(f'{os.path.join(self.base_dir, self.dataset_dir)} exists!')
        else:
            # download the dataset
            os.makedirs(os.path.join(self.base_dir, self.dataset_dir), exist_ok=True)
            os.chdir(os.path.join(self.base_dir, self.dataset_dir))
            os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz')
            os.system('gunzip covtype.data.gz')
            os.system('rm -f covtype.data.gz')
            os.chdir('../..')
            
        self.num_batch = 10
        self.n_total = 0
        with open(os.path.join(self.base_dir, self.dataset_dir, 'covtype.data'), 'r') as f:
            while 1:
                line = f.readline()
                # EOF
                if line == '':
                    break
                self.n_total += 1

        self.n_per_batch = int(self.n_total/self.num_batch)
        self.data = []
        self.target = []
        self.normalize= normalize
        self.mu = None
        self.std = None
        self.t = 0
        self.set_t(self.t)
        self.num_class = 7
        self.cate_feat = [[i] for i in range(10, 54)]

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return len(self.target) 
    
    def set_t(self, t):
        self.t = t
        self.data = []
        self.target = []
        # set starting and ending indices
        s = self.t * self.n_per_batch
        if self.t == self.num_batch - 1:
            e = self.n_total
        else:
            e = (self.t + 1) * self.n_per_batch

        with open(os.path.join(self.base_dir, self.dataset_dir, 'covtype.data'), 'r') as f:
            i = 0
            while 1:
                line = f.readline()
                if i >= e:
                    break
                if i >= s and i < e: 
                    a = line.split(',')
                    self.target.append(int(a[-1])-1)
                    l = []
                    for j in range(0, len(a)-1):
                        l.append(float(a[j]))
                    self.data.append(l)
                i += 1
        self.data = np.array(self.data, dtype='float32')
        self.target = np.array(self.target, dtype='int64')
        
        if t == 0:
            self.mu = np.mean(self.data[:, 0:10], axis=0)
            self.std = np.maximum(np.std(self.data[:, 0:10], axis=0), 1e-5)
            
        if self.normalize:
            self.data[:, 0:10] = (self.data[:, 0:10] - self.mu) / self.std

# KDD99 dataset
def one_hot_kdd(df, one_hot_features):
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
        df = one_hot_kdd(df, one_hot_features)
        f = lambda x: 0 if x == 'normal' else 1
        df['label'] = df['label'].map(f) 
        self.alldata = df.drop(['label'],axis=1).to_numpy()
        self.alltarget = df['label'].to_numpy()
        self.batch_data_num = int(self.alldata.shape[0] / self.num_batch)

        self.normalize_indices = [0,1,2,4,5,6,7,9,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
        self.mu = np.mean(self.alldata[:self.batch_data_num, self.normalize_indices], axis = 0)
        self.std = np.maximum(np.std(self.alldata[:self.batch_data_num, self.normalize_indices]), 1e-5)
        self.cate_feat = [[3], [8], [10], [11], [17], [18], list(range(38,41)), list(range(41,107)), list(range(107,118))]

    def set_t(self, t):
        self.t = t
        start = self.t * self.batch_data_num
        end = (self.t + 1) * self.batch_data_num
        self.data = self.alldata[start:end,:]
        self.target = self.alltarget[start:end]
        if self.normalize:
            self.data[:, self.normalize_indices] \
                = (self.data[:, self.normalize_indices] - self.mu) / self.std

# Electricity dataset

def one_hot_electricity(df, cat_feats):
    """Return dataframe after one-hot encoding
    Args
        df        -- the dataframe containing data
        cat_feats -- the categorical features
    """

    for feat in cat_feats:
        one_hot = pd.get_dummies(df[feat], prefix = feat)
        df = df.drop(columns = [feat])
        df = df.merge(one_hot, left_index = True, right_index = True)

    return df


class ElectricityDataset(Data.Dataset):

    def __init__(self, normalize = True, batch_days = 30):
        super(ElectricityDataset, self).__init__()
        self.base_dir = 'datasets'
        self.dataset_dir = 'electricity'
        if os.path.exists(os.path.join(self.base_dir, self.dataset_dir)):
            print(f'{os.path.join(self.base_dir, self.dataset_dir)} exists!')
            os.chdir(os.path.join(self.base_dir, self.dataset_dir))
        else:
            os.makedirs(os.path.join(self.base_dir, self.dataset_dir), exist_ok=True)
            os.chdir(os.path.join(self.base_dir, self.dataset_dir))
            os.system('wget https://datahub.io/machine-learning/electricity/r/electricity.csv')

        # self.num_batch = 943
        self.batch_days = batch_days

        self.all_data, self.data = [], []
        self.all_target, self.target = [], []
        self.generate('electricity.csv')

        os.chdir('../../')

        self.normalize = normalize
        self.t = 0
        self.set_t(self.t)
        self.num_class = 2
        self.cate_feat = [list(range(6, 13))]

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return self.target.shape[0]

    def generate(self, file_path):
        df = pd.read_csv(file_path)
        df['class'] = pd.factorize(df['class'])[0]
        df = one_hot_electricity(df, ['day'])

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

        self.num_batch = len(self.all_data) // self.batch_days

        # numeric features: [0, 1, 2, 3, 4, 5]
        self.normalize_indices = [0, 1, 2, 3, 4, 5]
        self.mu = np.mean(self.all_data[0][:, self.normalize_indices], axis = 0)
        self.std = np.maximum(np.std(self.all_data[0][:, self.normalize_indices]), 1e-5)

        return

    def set_t(self, t):
        self.t = t
        self.data = np.concatenate(
            self.all_data[t * self.batch_days : (t + 1) * self.batch_days],
            axis = 0).astype('float32')

        self.target = np.concatenate(
            self.all_target[t * self.batch_days : (t + 1) * self.batch_days],
            axis = 0).astype('float32')

        self.target = np.array(self.target, dtype = 'int64')

        if self.normalize:
            self.data[:, self.normalize_indices] \
                = (self.data[:, self.normalize_indices] - self.mu) / self.std

        return

# ONP dataset
class ONPDataset(Data.Dataset):
    def __init__(self, normalize=True):
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
        self.cate_feat = [[11], [12], [13], [14], [15], [16], list(range(29, 36)), [36]]

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

# California housing dataset
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

class california_housing_dataset(Data.Dataset):
    def __init__(self, num_batch=21, percentile=[10, 90], variance_coeff=0.5, normalize=True):
        super(california_housing_dataset, self).__init__()
        self.num_batch = num_batch
        self.real_data = fetch_california_housing().data
        self.all_data = []   #[data0, data1, ...]
        self.data = []
        self.real_target = fetch_california_housing().target
        self.all_target = []
        self.target = []
        self.dim = self.real_data.shape[1]

        self.target_range = [
            np.percentile(np.sort(self.real_target), percentile[0]),
            np.percentile(np.sort(self.real_target), percentile[1])
        ]
        self.thresholds = get_threshold(self.target_range, variance_coeff, self.num_batch-1)
        
        self.n_sample = self.real_target.shape[0]
        
        self.num_class = 2
        self.normalize = normalize
        self.num_batch = num_batch
        self.base_dir = None
        self.dataset_dir = None

        self.preprocessing()

        self.t = 0
        self.set_t(self.t)
        self.cate_feat = []
        

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return int(self.n_sample * 0.7)

    # sklearn.datasets.fetch_california_housing dataset, all features is real number 
    def preprocessing(self):
        for t in range(self.num_batch):
            indices = np.random.choice(range(self.n_sample), int(self.n_sample * 0.7), replace=False)

            data = np.array(self.real_data[indices])
            target = np.array([self.real_target[i] >= self.thresholds[t] for i in list(indices)], dtype="int64") 
        
            if t == 0:
                self.mu = np.mean(data, axis=0)
                self.std = np.maximum(np.std(data), 1e-5)
            elif self.normalize == True:
                data = (data - self.mu) / self.std 

            self.all_data.append(data)
            self.all_target.append(target)
    
    def set_t(self, t):
        self.t = t
        self.data = self.all_data[t]
        self.target = self.all_target[t]

# Wine dataset (white/red)     
class wine_quality_dataset(Data.Dataset):
    def __init__(self, dataset='white', normalize=True):
        super(wine_quality_dataset, self).__init__()
        assert dataset in ['white', 'red']
        
        self.base_dir = 'datasets'
        self.dataset_dir = 'wine_quality'
        if os.path.exists(os.path.join(self.base_dir, self.dataset_dir)):
            print(f'{os.path.join(self.base_dir, self.dataset_dir)} exists!')
        else:
            # download the dataset
            os.makedirs(os.path.join(self.base_dir, self.dataset_dir), exist_ok=True)
            os.chdir(os.path.join(self.base_dir, self.dataset_dir))
            os.system(f'wget http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv')
            os.system(f'wget http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')
            os.chdir('../..')  
            
        self.df = pd.read_csv(os.path.join(self.base_dir, self.dataset_dir, f'winequality-{dataset}.csv'), sep=';', dtype=float) 
        self.X = self.df.iloc[:, :-1].to_numpy()
        self.y = self.df.iloc[:, -1].to_numpy()    
        self.n_sample = self.X.shape[0]
        self.normalize_indices = list(range(self.X.shape[1]))
            
        self.thresholds = [4,5,6,7,8] 
        self.sample_rate = 0.7
        self.num_batch = len(self.thresholds)
        self.data = []
        self.target = []
        self.normalize= normalize
        self.mu = None
        self.std = None
        self.t = 0
        self.set_t(self.t)
        self.num_class = 2
        self.cate_feat = []
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return int(self.n_sample*self.sample_rate) 
    
    def set_t(self, t):
        self.t = t
        sample_indexes = random.sample(range(self.n_sample), int(self.n_sample*self.sample_rate)) 
        self.data = self.X[sample_indexes,:]
        self.target = self.y[sample_indexes]
        self.target = [1 if val >= self.thresholds[self.t] else 0 for val in self.target]
        
        if t == 0:
            self.mu = np.mean(self.data[:, self.normalize_indices], axis=0)
            self.std = np.maximum(np.std(self.data[:, self.normalize_indices], axis=0), 1e-5)
            
        if self.normalize:
            self.data[:, self.normalize_indices] = (self.data[:, self.normalize_indices] - self.mu) / self.std

# Power plant
class CCPP_dataset(Data.Dataset):  
    def __init__(self, k, normalize=True):
        super(CCPP_dataset, self).__init__()
        
        self.base_dir = 'datasets'
        self.dataset_dir = 'CCPP'
        if os.path.exists(os.path.join(self.base_dir, self.dataset_dir)):
            print(f'{os.path.join(self.base_dir, self.dataset_dir)} exists!')
        else:
            # download the dataset
            os.makedirs(self.base_dir, exist_ok=True)
            os.chdir(self.base_dir)
            os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip')
            os.system('unzip CCPP.zip')
            os.system('rm -f CCPP.zip')
            os.chdir('..')

        self.df = pd.read_excel(os.path.join(self.base_dir, self.dataset_dir, 'Folds5x2_pp.xlsx'), engine='openpyxl', sheet_name='Sheet1')
        self.X = self.df.iloc[:, :-1].to_numpy()
        self.y = self.df.iloc[:, -1].to_numpy()    
        self.n_sample = self.X.shape[0]
        self.normalize_indices = list(range(self.X.shape[1]))
        
        THs = np.percentile(self.y, 10)
        THe = np.percentile(self.y, 90)
        k = k         
        self.thresholds = []
        u = (THe - THs) / k
        variant_step = np.random.normal(u, u*0.5, k+1)
        variant_step[0] = 0 # no need to move at starting threshold
        t = THs
        for i in range(k+1):
            t += variant_step[i]
            self.thresholds.append(t)
            
        self.num_batch = len(self.thresholds)
        self.sample_rate = 0.7
        self.data = []
        self.target = []
        self.normalize= normalize
        self.mu = None
        self.std = None
        self.t = 0
        self.set_t(self.t)
        self.num_class = 2
        self.cate_feat = []
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return int(self.n_sample*self.sample_rate) 
    
    def set_t(self, t):
        self.t = t
        sample_indexes = random.sample(range(self.n_sample), int(self.n_sample*self.sample_rate))
        self.data = self.X[sample_indexes,:]
        self.target = self.y[sample_indexes]
        self.target = [1 if val >= self.thresholds[self.t] else 0 for val in self.target]
        
        if t == 0:
            self.mu = np.mean(self.data[:, self.normalize_indices], axis=0)
            self.std = np.maximum(np.std(self.data[:, self.normalize_indices], axis=0), 1e-5)
            
        if self.normalize:
            self.data[:, self.normalize_indices] = (self.data[:, self.normalize_indices] - self.mu) / self.std

# Ford price
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
        self.base_dir = 'datasets'
        self.dataset_dir = 'ford_price'
        self.normalize = normalize
        self.normalize_indices = [0, 1, 2, 3]
        self.cate_feat = [list(range(4, 41))]

        dataset_path = os.path.join(self.base_dir, self.dataset_dir)
        if os.path.exists(os.path.join(self.base_dir, self.dataset_dir)):
            print(f'{os.path.join(self.base_dir, self.dataset_dir)} exists!')
        else:
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
            

dataset_dict = {'translate':TranslateDataset(),
                'rotate':RotateDataset(),
                'ball':HyperballDataset(),
                'house':california_housing_dataset(num_batch=11, normalize=True),
                'wine_white':wine_quality_dataset(dataset='white', normalize=True),
                'wine_red':wine_quality_dataset(dataset='red', normalize=True),
                'power':CCPP_dataset(k=10, normalize=True),
                'price':ford_price_dataset(time_slice=10, normalize=True),
                'gas':GasSensorDataset(normalize=True),
                'covertype':CovertypeDataset(normalize=True),
                'kdd':KDD99Dataset(normalize=True),
                'electricity':ElectricityDataset(normalize=False), #The dataset has already been normalized originally
                'onp':ONPDataset(normalize=True)}