import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np

import copy
import random
import os
import argparse

from utils import train, test, train_soft, test_soft, train_dynamic, test_dynamic, predict_dynamic, train_cost_sensitive, test_cost_sensitive
from models import MLP, DynamicPredictor
from datasets import BufferDataset, SoftmaxDataset

from datasets import GasSensorDataset, CovertypeDataset


def set_seed(seed):
    '''
    fixed random seed
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
def initialize(trainset, opt, F, optimizer, data_softmax_history, device):
    '''
    train data_0 and get softmax_0 on data_0 
    '''
    trainset.set_t(0)
    baseset = copy.deepcopy(trainset)
    
    data_loader = Data.DataLoader(trainset, batch_size=opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    for i in range(opt.epochs):
        train(data_loader, F, optimizer, device)
        loss, acc = test(data_loader, F, device)
        print(f'[Train 0] loss:{loss}, acc:{acc}', file=opt.output_file, flush=True)

    data_loader = Data.DataLoader(baseset, batch_size=opt.batch_size,
                                      shuffle=False, num_workers=opt.num_workers)
    _, _, log = test(data_loader, F, device, return_softmax=True)
    data_softmax_history.append(log)

    return baseset


def test_data_t(trainset, opt, F, test_acc, t, device):
    '''
    test data_t
    '''
    data_loader = Data.DataLoader(trainset, batch_size=opt.batch_size,
                                  shuffle=False, num_workers=opt.num_workers)
    loss, acc = test(data_loader, F, device)
    print(f'[Test {t}] loss:{loss}, acc:{acc}', file=opt.output_file, flush=True)
    test_acc.append(acc)

    
def train_data_t(trainset, opt, F, optimizer, t, device):
    '''
    train data_t
    '''
    data_loader = Data.DataLoader(trainset, batch_size=opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    for i in range(opt.epochs):
        train(data_loader, F, optimizer, device)
        loss, acc = test(data_loader, F, device)
        print(f'[Train {t}] loss:{loss}, acc:{acc}', file=opt.output_file, flush=True)

    
def get_softmax_data_0(baseset, opt, F, data_softmax_history, device):
    '''
    get softmax_t on data_0
    '''
    data_loader = Data.DataLoader(baseset, batch_size=opt.batch_size,
                                  shuffle=False, num_workers=opt.num_workers)
    _, _, log = test(data_loader, F, device, return_softmax=True)
    data_softmax_history.append(log)


def train_dynamic_predictor(opt, DP, d_optimizer, data_softmax_history, t, device):
    '''
    train dynamic predictor {softmax_(t-w),...,softmax_(t-2), softmax_(t-1)} --> softmax_t
    '''
    softmax_dataset = SoftmaxDataset(data_softmax_history, mode='train')
    softmax_data_loader = Data.DataLoader(softmax_dataset, batch_size=opt.batch_size,
                                          shuffle=True, num_workers=opt.num_workers)
    for i in range(opt.d_epochs):
        train_dynamic(softmax_data_loader, DP, d_optimizer, device)
        loss = test_dynamic(softmax_data_loader, DP, device)
        print(f'[Train dynamic predictor {t}] loss:{loss}', file=opt.output_file, flush=True)

    if len(data_softmax_history) > opt.time_window:
        data_softmax_history.pop(0)

        
def pred_next_softmax_on_data_0(opt, DP, data_softmax_history, device):
    '''
    predict softmax_(t+1) on data_0 {softmax_(t-w+1),...,softmax_(t-1), softmax_(t)} --> predicted_softmax_(t+1)
    '''
    softmax_dataset = SoftmaxDataset(data_softmax_history, mode='test')
    softmax_data_loader = Data.DataLoader(softmax_dataset, batch_size=opt.batch_size,
                                          shuffle=False, num_workers=opt.num_workers)
    pred_next_softmax = predict_dynamic(softmax_data_loader, DP, device)

    return pred_next_softmax


def predict_next_decision_boundary(baseset, opt, F, optimizer, t, pred_next_softmax, device):
    '''
    predict decision boundary t+1 by training data_0 --> predicted_softmax_(t+1)
    '''
    print(f'[Predict decision boundary {t+1}]', end='', file=opt.output_file, flush=True)
    if opt.method == 'soft':
        soft_baseset = BufferDataset(baseset.data, pred_next_softmax, target_type='soft')
        data_loader = Data.DataLoader(soft_baseset, batch_size=opt.batch_size,
                                              shuffle=True, num_workers=opt.num_workers)
        for i in range(opt.epochs):
            train_soft(data_loader, F, optimizer, device)
            loss = test_soft(data_loader, F, device)
            print(f'loss:{loss}', file=opt.output_file, flush=True)
    elif opt.method == 'hard':
        y = [np.argmax(s) for s in np.array(pred_next_softmax)]
        y = np.array(y, dtype='int64')
        baseset.target = y
        data_loader = Data.DataLoader(baseset, batch_size=opt.batch_size,
                                      shuffle=True, num_workers=opt.num_workers)
        for i in range(opt.epochs):
            train(data_loader, F, optimizer, device)
            loss, acc = test(data_loader, F, device)
            print(f'loss:{loss}, acc:{acc}', file=opt.output_file, flush=True)
    else:
        soft_baseset = BufferDataset(baseset.data, pred_next_softmax, target_type='soft')
        data_loader = Data.DataLoader(soft_baseset, batch_size=opt.batch_size, 
                                      shuffle=True, num_workers=opt.num_workers)
        for i in range(opt.epochs):
            train_cost_sensitive(data_loader, F, optimizer, device)
            loss, acc = test_cost_sensitive(data_loader, F, device)
            print(f'loss:{loss}, acc:{acc}', file=opt.output_file, flush=True)


def two_stage_method(opt, trainset, device):        
    '''
    implement two stage method
    '''
    F = MLP(in_size = trainset.dim, out_size = opt.class_num).to(device)
    DP = DynamicPredictor(in_size = opt.class_num, d_model = trainset.dim * 2).to(device)
    optimizer = optim.Adam(F.parameters(), lr=opt.lr, weight_decay = opt.decay)
    d_optimizer = optim.AdamW(DP.parameters(), lr=opt.d_lr, weight_decay = opt.d_decay)
    
    data_softmax_history = []
    test_acc = []

    baseset = initialize(trainset, opt, F, optimizer, data_softmax_history, device) # train data_0 get softmax_0 on data_0 
    
    for t in range(1, trainset.time_slice):    
        trainset.set_t(t)

        test_data_t(trainset, opt, F, test_acc, t, device)  # test data t
        train_data_t(trainset, opt, F, optimizer, t, device) # train data t

        if t == trainset.time_slice - 1:
            break

        get_softmax_data_0(baseset, opt, F, data_softmax_history, device) # get softmax_t on data_0
        train_dynamic_predictor(opt, DP, d_optimizer, data_softmax_history, t, device) # train dynamic predictor

        if opt.method == 'finetune':
            continue

        if t >= opt.activate_dynamic_t:
            pred_next_softmax = pred_next_softmax_on_data_0(opt, DP, data_softmax_history, device) # predict softmax_(t+1) on data_0 
            predict_next_decision_boundary(baseset, opt, F, optimizer, t, pred_next_softmax, device) # predict decision boundary t+1 by training data_0
            
    print('Test acc log:', test_acc, file=opt.output_file, flush=True)
    print('Mean acc:', sum(test_acc)/len(test_acc), file=opt.output_file, flush=True)
    
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-lr', type=float, default=2e-3)
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-decay', type=float, default=5e-5)
    parser.add_argument('-d_lr', type=float, default=1e-3)
    parser.add_argument('-d_epochs', type=int, default=50)
    parser.add_argument('-d_decay', type=float, default=0)
    parser.add_argument('-method', type=str, default='finetune')
    parser.add_argument('-activate_dynamic_t', type=int, default=3)
    parser.add_argument('-time_window', type=int, default=3)
    parser.add_argument('-time_slice', type=int, default=21)
    parser.add_argument('-class_num', type=int, default=2)

    parser.add_argument('-output_file', type=str, default=None)
    
    opt = parser.parse_args()
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if opt.output_file == None:
        opt.output_file = f'logs/{opt.method}_tw{opt.time_window}.log'
    opt.output_file = open(opt.output_file, 'w')
        
    set_seed(opt.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using', device, file=opt.output_file, flush=True)

    trainset = GasSensorDataset()    
    two_stage_method(opt, trainset, device)

    
if __name__ == '__main__':        
    main()
    
