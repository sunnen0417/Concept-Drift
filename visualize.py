from datasets import dataset_dict
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from experiment_batch0 import set_seed
from sklearn.decomposition import PCA
import os

def visualize(X, y, colors, save_img_path):
    plt.figure(figsize=(5, 5))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis([-5, 5, -5, 5])
    for i in range(len(colors)):
        mask = y == i
        plt.plot(X[mask][:, 0], X[mask][:, 1], f'{colors[i]}.', label=f'class {i}')
    plt.legend(loc='upper right')
    plt.savefig(save_img_path)
        

if __name__ == '__main__':
    trainset = dataset_dict['revolution']
    dataset_name = 'revolution'
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colors = colors[0:trainset.num_class]
    dir_name = f'{dataset_name}_visualize'
    os.makedirs(dir_name, exist_ok=True)
    dim = len(trainset.data[0])
    if dim > 2:
        pca = PCA(n_components=2)
    
    for t in range(trainset.num_batch):
        trainset.set_t(t)
        if dim > 2:
            if t == 0:
                X = pca.fit_transform(trainset.data)
            else:
                X = pca.transform(trainset.data)
        else:
            X = trainset.data
        y = trainset.target
        save_img_path = os.path.join(dir_name, f'{dataset_name}_batch{t}_visualize.png')
        visualize(X, y, colors, save_img_path)
