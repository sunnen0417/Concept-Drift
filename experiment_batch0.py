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
from datasets import ToyDataset, SoftmaxDataset, BufferDataset, get_translate_concept, get_rotate_concept, get_uncertain_translate_concept
from datasets import GasSensorDataset, CovertypeDataset 
from utils import train, test, train_soft, test_soft, train_dynamic, test_dynamic, predict_dynamic, train_cost_sensitive, test_cost_sensitive, draw_decision_boundary
from models import Classifier, MLP, DynamicPredictor
from hyper_ball import get_hyperball_concept_translate, HyperBallData

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # Hyperparameters
    seed = 0
    set_seed(seed)
    #img_dir = 'gas_finetune_10'
    last_step_method = 'soft' #'hard', 'soft', 'cost'
    #os.makedirs(img_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using', device)
    num_workers = 0
    batch_size = 64
    lr = 2e-3
    epochs = 50
    decay = 5e-5
    d_lr = 1e-3
    d_epochs = 50
    d_decay = 0
    #x_range = [-5, 5]
    #y_range = [-5, 5]
    #n_sample = 4000
    activate_dynamic_t = 3 # set this to a very large value if finetune only (no dynamic prediction)
    time_window = 3
    """ 
    # translate
    s = [-1, 4, -20]
    e = [-1, 4, 20]
    slice = 20
    concept = get_uncertain_translate_concept(s, e, slice, std_ratio=0.5)   
    #rotate
    rpc = math.pi / 20
    unit = 81
    angles = []
    for i in range(unit):
        angles.append(i*rpc)
    concept = get_rotate_concept(angles)
    """
    #hyperball
    #concept = get_hyperball_concept_translate(r_range=[[20,15],[2,30]], c_range=[[0,20],[0,20]], K_range=[1, 1], t = 21)
    
    # training set
    #trainset = ToyDataset(x_range, y_range, n_sample, concept)
    #trainset = HyperBallData(ranges=[[x_range[0],y_range[0]],[x_range[1], y_range[1]]], n_sample=n_sample, concept=concept, noise=0)
    #concept = concept[0]
    trainset = GasSensorDataset(normalize=True)
    concept = np.arange(trainset.concept)
    dim = trainset.data.shape[1]
    classes = 6
    
    # Classifier
    F = MLP(in_size=dim, out_size=classes).to(device)
    optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)
    # Dynamic predictor
    DP = DynamicPredictor(in_size=classes, d_model=classes*2).to(device)
    d_optimizer = optim.AdamW(DP.parameters(), lr=d_lr, weight_decay=d_decay)
    
    # time_window * n_sample * dim
    data_softmax_history = []
    test_acc = []
    for t in range(len(concept)):
        trainset.set_t(t)
        if t > 0:
            print(f'Test {t}')
            data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                    shuffle=False, num_workers=num_workers)
            loss, acc = test(data_loader, F, device)
            print(f'loss:{loss}, acc:{acc}')
            test_acc.append(acc)
        print(f'Train {t}')
        # keep batch 0 dataset
        if t == 0:
            baseset = copy.deepcopy(trainset)
        data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers)
        for i in range(epochs):
            print('Epoch:', i+1)
            train(data_loader, F, optimizer, device)
            loss, acc = test(data_loader, F, device)
            print(f'loss:{loss}, acc:{acc}')
        
        if t > 0:
            #draw_decision_boundary(data_loader, F, device, x_range=x_range, y_range=y_range, newfig=False, db_color='g')
            #plt.savefig(f'{img_dir}/concept{t-1}.png')
            if t == len(concept) - 1:
                break
        
        #draw_decision_boundary(data_loader, F, device, x_range=x_range, y_range=y_range, newfig=True, db_color='b')            
        
        print(f'Get data softmax {t}')
        data_loader = Data.DataLoader(baseset, batch_size=batch_size, 
                                    shuffle=False, num_workers=num_workers)
        _, _, log = test(data_loader, F, device, return_softmax=True)
        data_softmax_history.append(log)
        
        if t > 0:
            print(f'Train dynamic predictor {t}')
            softmax_dataset = SoftmaxDataset(data_softmax_history, mode='train')
            softmax_data_loader = Data.DataLoader(softmax_dataset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=num_workers)
            for i in range(d_epochs):
                print('Epoch:', i+1)
                train_dynamic(softmax_data_loader, DP, d_optimizer, device)
                loss = test_dynamic(softmax_data_loader, DP, device)
                print(f'loss:{loss}')
        
        if len(data_softmax_history) > time_window:
            data_softmax_history.pop(0)
            
        if t >= activate_dynamic_t:
            print(f'Predict data softmax {t+1}')
            softmax_dataset = SoftmaxDataset(data_softmax_history, mode='test')
            softmax_data_loader = Data.DataLoader(softmax_dataset, batch_size=batch_size, 
                                                  shuffle=False, num_workers=num_workers)
            pred_next_softmax = predict_dynamic(softmax_data_loader, DP, device)
        
            print(f'Predict decision boundary {t+1}')
            if last_step_method == 'soft':
                soft_baseset = BufferDataset(baseset.data, pred_next_softmax, target_type='soft')
                data_loader = Data.DataLoader(soft_baseset, batch_size=batch_size, 
                                              shuffle=True, num_workers=num_workers)
                for i in range(epochs):
                    print('Epoch:', i+1)
                    train_soft(data_loader, F, optimizer, device)
                    loss = test_soft(data_loader, F, device)
                    print(f'loss:{loss}')
            elif last_step_method == 'hard':
                y = [np.argmax(s) for s in np.array(pred_next_softmax)] 
                y = np.array(y, dtype='int64')
                baseset.target = y
                data_loader = Data.DataLoader(baseset, batch_size=batch_size, 
                                              shuffle=True, num_workers=num_workers)
                for i in range(epochs):
                    print('Epoch:', i+1)
                    train(data_loader, F, optimizer, device)
                    loss, acc = test(data_loader, F, device)
                    print(f'loss:{loss}, acc:{acc}')
            elif last_step_method == 'cost':
                soft_baseset = BufferDataset(baseset.data, pred_next_softmax, target_type='soft')
                data_loader = Data.DataLoader(soft_baseset, batch_size=batch_size, 
                                              shuffle=True, num_workers=num_workers)
                for i in range(epochs):
                    print('Epoch:', i+1)
                    train_cost_sensitive(data_loader, F, optimizer, device)
                    loss, acc = test_cost_sensitive(data_loader, F, device)
                    print(f'loss:{loss}, acc:{acc}')               
                
        data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers)        
        #draw_decision_boundary(data_loader, F, device, x_range=x_range, y_range=y_range, newfig=False, db_color='r')
    print('Test acc log:', test_acc)
    print('Mean acc:', sum(test_acc)/len(test_acc))

        
        
    
