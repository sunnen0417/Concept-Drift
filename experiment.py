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
    method = 'finetune'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using', device)
    num_workers = 0
    batch_size = 256
    lr = 2e-3
    epochs = 50 
    decay = 5e-5
    d_lr = 1e-3 
    d_epochs = 50  
    d_decay = 0
    activate_dynamic_t = 100
    time_window = 3


    for finetune, method, activate_dynamic_t in [(True, '_', 100),(False, 'hard', 3), (False, 'soft', 3), (False, 'cost_sens', 3)]:
        set_seed(0) 

        trainset = GasSensorDataset(normalize=True) #CovertypeDataset(normalize=True)
        in_size, out_size = 128, 6     # 54, 7
        d_model = 14
        
        concept = np.arange(trainset.concept)
        
        F = MLP(in_size=in_size, out_size=out_size).to(device) 
        DP = DynamicPredictor(in_size=out_size, d_model=d_model).to(device)
        optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)
        d_optimizer = optim.AdamW(DP.parameters(), lr=d_lr, weight_decay=d_decay)
        # time_window * n_sample * dim
        data_softmax_history = []
        test_acc = []
        previous_classifiers_lst = []  #####
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
            if t == 0:       
                baseset = copy.deepcopy(trainset)     ##### 
            

            ##### (1) trainset: only batch t
            data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                          shuffle=True, num_workers=num_workers)     
            ##### 
            
            # add data in batch t to baseset
            if t != 0: 
                baseset.concat_dataset(copy.deepcopy(trainset))  
            print('baseset.shape', baseset.data.shape, baseset.target.shape)
            
            ##### (2) or all batches (label=GT label)
            #data_loader = Data.DataLoader(baseset, batch_size=batch_size, 
            #                            shuffle=True, num_workers=num_workers)      
            #####
            
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
            
            ##### store classifiers
            previous_classifiers_lst.append(copy.deepcopy(F))
            if len(previous_classifiers_lst) > time_window+1:
                previous_classifiers_lst.pop(0)
            print('len(previous_classifiers_lst)', len(previous_classifiers_lst))
            
            
            #draw_decision_boundary(data_loader, F, device, x_range=x_range, y_range=y_range, newfig=True, db_color='b')            
            
            print(f'Get data softmax {t}')
            data_loader = Data.DataLoader(baseset, batch_size=batch_size, 
                                        shuffle=False, num_workers=num_workers)   ##### should be all batches
            
            
            ##### 
            #_, _, log = test(data_loader, F, device, return_softmax=True)
            #data_softmax_history.append(log)   
            data_softmax_history = []
            for F_previous in previous_classifiers_lst: 
                _, _, log = test(data_loader, F_previous, device, return_softmax=True)  
                data_softmax_history.append(log) 
                print(log[0])
                print('len(data_softmax_history)', len(data_softmax_history)) 
            #####
            
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
                if method == 'soft':
                    soft_baseset = BufferDataset(baseset.data, pred_next_softmax, target_type='soft')
                    data_loader = Data.DataLoader(soft_baseset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=num_workers)
                    for i in range(epochs):
                        print('Epoch:', i+1)
                        train_soft(data_loader, F, optimizer, device)
                        loss = test_soft(data_loader, F, device)
                        print(f'loss:{loss}')
                elif method == 'hard':
                    y = [np.argmax(s) for s in np.array(pred_next_softmax)] 
                    y = np.array(y, dtype='int64')
                    #baseset.target = y
                    #data_loader = Data.DataLoader(baseset, batch_size=batch_size, 
                    #                              shuffle=True, num_workers=num_workers)
                    hard_baseset = BufferDataset(baseset.data, y, target_type='hard')
                    data_loader = Data.DataLoader(hard_baseset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=num_workers)
                    #####
                    for i in range(epochs):
                        print('Epoch:', i+1)
                        train(data_loader, F, optimizer, device)
                        loss, acc = test(data_loader, F, device)
                        print(f'loss:{loss}, acc:{acc}')
                else:
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

            
        
    
