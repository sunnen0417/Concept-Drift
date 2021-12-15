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
from utils import train, test, train_soft, test_soft, train_dynamic, test_dynamic, predict_dynamic, train_cost_sensitive, test_cost_sensitive, draw_decision_boundary, sample, update_all_vaes
from models import Classifier, MLP, DynamicPredictor, VAE
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
    #img_dir = 'gas_dynamic_soft_10'
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
    vae_lr = 1e-3
    vae_epochs = 250
    vae_decay = 0
    theta = 0.1
    sample_n = 200
    eps = 0.1
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
    # Dynamic Predictor
    DP = DynamicPredictor(in_size=classes, d_model=classes*2).to(device)
    d_optimizer = optim.AdamW(DP.parameters(), lr=d_lr, weight_decay=d_decay)
    # VAE
    vae_list = []
    vae_optimizer_list = []
    for i in range(classes):
        vae_list.append(VAE(feat_size=dim, hidden_size=128, latent_size=16))
        vae_optimizer_list.append(optim.Adam(vae_list[i].parameters(), lr=vae_lr, weight_decay=vae_decay))
    
    classifier_list = []
    test_acc = []
    for t in range(len(concept)):
        trainset.set_t(t)
        data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)
        if t > 0:
            print(f'Test {t}')
            loss, acc = test(data_loader, F, device)
            print(f'loss:{loss}, acc:{acc}')
            test_acc.append(acc)
            
        print(f'Update VAEs {t}')    
        update_all_vaes(data_loader, vae_list, vae_optimizer_list, vae_epochs, theta, sample_n, eps, device)
        
        print(f'Sample data from VAEs {t}')
        aug_data = []
        aug_label = []
        for i in range(classes):
            data = sample(vae_list[i], theta, sample_n, device, batch_size=batch_size, num_workers=num_workers)
            labels = np.full(len(data), i)
            aug_data += data.tolist()
            aug_label += labels.tolist()
        aug_data += trainset.data.tolist()
        aug_label += trainset.target.tolist()
        aug_trainset = BufferDataset(aug_data, aug_label, target_type='hard')
        
        print(f'Train {t}')
        data_loader = Data.DataLoader(aug_trainset, batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers)
        for i in range(epochs):
            print('Epoch:', i+1)
            train(data_loader, F, optimizer, device)
            loss, acc = test(data_loader, F, device)
            print(f'loss:{loss}, acc:{acc}')
        
        classifier_list.append(copy.deepcopy(F))
        
        if t > 0:
            #draw_decision_boundary(data_loader, F, device, x_range=x_range, y_range=y_range, newfig=False, db_color='g')
            #plt.savefig(f'{img_dir}/concept{t-1}.png')
            if t == len(concept) - 1:
                break
        
        #draw_decision_boundary(data_loader, F, device, x_range=x_range, y_range=y_range, newfig=True, db_color='b')            
        
        print(f'Get data softmax {t}')
        data_loader = Data.DataLoader(aug_trainset, batch_size=batch_size, 
                                    shuffle=False, num_workers=num_workers)
        # time_window * n_sample * dim
        data_softmax_history = []
        for i in range(len(classifier_list)):
            _, _, log = test(data_loader, classifier_list[i], device, return_softmax=True)
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
        
        if len(classifier_list) > time_window:
            classifier_list.pop(0)
            data_softmax_history.pop(0)
            
        if t >= activate_dynamic_t:
            print(f'Predict data softmax {t+1}')
            softmax_dataset = SoftmaxDataset(data_softmax_history, mode='test')
            softmax_data_loader = Data.DataLoader(softmax_dataset, batch_size=batch_size, 
                                                  shuffle=False, num_workers=num_workers)
            pred_next_softmax = predict_dynamic(softmax_data_loader, DP, device)
        
            print(f'Predict decision boundary {t+1}')
            if last_step_method == 'soft':
                aug_trainset = BufferDataset(aug_data, pred_next_softmax, target_type='soft')
                data_loader = Data.DataLoader(aug_trainset, batch_size=batch_size, 
                                              shuffle=True, num_workers=num_workers)
                for i in range(epochs):
                    print('Epoch:', i+1)
                    train_soft(data_loader, F, optimizer, device)
                    loss = test_soft(data_loader, F, device)
                    print(f'loss:{loss}')
            elif last_step_method == 'hard':
                y = [np.argmax(s) for s in np.array(pred_next_softmax)] 
                y = np.array(y, dtype='int64')
                aug_trainset = BufferDataset(aug_data, y, target_type='hard')
                data_loader = Data.DataLoader(aug_trainset, batch_size=batch_size, 
                                              shuffle=True, num_workers=num_workers)
                for i in range(epochs):
                    print('Epoch:', i+1)
                    train(data_loader, F, optimizer, device)
                    loss, acc = test(data_loader, F, device)
                    print(f'loss:{loss}, acc:{acc}')
            elif last_step_method == 'cost':
                aug_trainset = BufferDataset(aug_data, pred_next_softmax, target_type='soft')
                data_loader = Data.DataLoader(aug_trainset, batch_size=batch_size, 
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

        
        
    
