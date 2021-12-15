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

# Custom dataset
class ToyDataset(Data.Dataset):
    def __init__(self, x_range, y_range, n_sample, concept):
        super(ToyDataset, self).__init__()
        self.n_sample = n_sample
        self.x_range = x_range
        self.y_range = y_range
        self.concept = np.array(concept)
        self.t = 0
        self.data = []
        self.target = []
        self.set_t(self.t)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return len(self.target)
    
    def set_t(self, t):
        self.t = t
        self.data = np.random.uniform([self.x_range[0], self.y_range[0]], [self.x_range[1], self.y_range[1]],
                                 size=(self.n_sample, 2))
        w = self.concept[t].reshape((-1, 1))
        aug_data = np.concatenate((self.data, np.ones((len(self.data), 1))), axis=1)
        y = np.matmul(aug_data, w).reshape((-1,))
        dist = np.absolute(y) / np.linalg.norm(self.concept[t])
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

# Gas sensor dataset
class GasSensorDataset(Data.Dataset):
    def __init__(self, normalize=False):
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
            
        self.concept = 10
        self.data = []
        self.target = []
        self.normalize= normalize
        self.mu = None
        self.std = None
        self.t = 0
        self.set_t(self.t)
        
            
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
    def __init__(self, normalize=False):
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
            
        self.concept = 10
        self.n_total = 0
        with open(os.path.join(self.base_dir, self.dataset_dir, 'covtype.data'), 'r') as f:
            while 1:
                line = f.readline()
                # EOF
                if line == '':
                    break
                self.n_total += 1

        self.n_per_concept = int(self.n_total/self.concept)
        self.data = []
        self.target = []
        self.normalize= normalize
        self.mu = None
        self.std = None
        self.t = 0
        self.set_t(self.t)
        
            
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return len(self.target) 
    
    def set_t(self, t):
        self.t = t
        self.data = []
        self.target = []
        # set starting and ending indices
        s = self.t * self.n_per_concept
        if self.t == self.concept - 1:
            e = self.n_total
        else:
            e = (self.t + 1) * self.n_per_concept

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