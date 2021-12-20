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
            
dataset_dict = {'translate':TranslateDataset,
                'rotate':RotateDataset,
                'ball':HyperballDataset,
                'gas':GasSensorDataset,
                'covertype':CovertypeDataset}