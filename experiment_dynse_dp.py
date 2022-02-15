import numpy as np
import copy
import os
import random
import torch
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from datasets import SoftmaxDataset, SoftmaxOnlineDataset, BufferDataset
from datasets import dataset_dict
from utils import train, test, train_soft, test_soft, train_dynamic, test_dynamic, predict_dynamic, train_cost_sensitive, test_cost_sensitive, draw_decision_boundary, split_train_valid
from models import LogisticRegression, MLP, DynamicPredictor, SubspaceBuffer, DynseDP
import argparse

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--lr', type=float, default=2e-3)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-dc', '--decay', type=float, default=5e-5)
    parser.add_argument('-dlr', '--d_lr', type=float, default=1e-3)
    parser.add_argument('-de', '--d_epochs', type=int, default=50)
    parser.add_argument('-ddc', '--d_decay', type=float, default=0)
    parser.add_argument('-tw', '--time_window', type=int, default=3)
    parser.add_argument('-mes', '--max_ensemble_size', type=int, default=25)
    parser.add_argument('-mbs', '--max_buffer_size', type=int, default=4)
    parser.add_argument('-k', '--k', type=int, default=5)
    parser.add_argument('-mc', '--max_centroids', type=int, default=20)
    parser.add_argument('-mi', '--max_instances', type=int, default=100)
    parser.add_argument('-d', '--dataset', type=str, default='translate') # see dataset_dict in datasets.py
    parser.add_argument('-c', '--classifier', type=str, default='lr') # 'lr':logistic regreesion, 'mlp':neural network
    parser.add_argument('-dv', '--device', type=str, default='cuda:0')
    parser.add_argument('-tr', '--train_ratio', type=float, default=0.8)
    parser.add_argument('-p', '--patience', type=int, default=7)
    return parser

if __name__ == '__main__':
    # Get arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Hyperparameters
    seed = args.seed
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('using', device)
    num_workers = args.num_workers
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    decay = args.decay
    d_lr = args.d_lr
    d_epochs = args.d_epochs
    d_decay = args.d_decay
    time_window = args.time_window
    max_ensemble_size = args.max_ensemble_size
    max_buffer_size = args.max_buffer_size
    k = args.k
    max_centroids = args.max_centroids
    max_instances = args.max_instances
    train_ratio = args.train_ratio
    patience = args.patience
    
    # Dataset
    trainset = dataset_dict[args.dataset]
    num_batch = trainset.num_batch
    dim = trainset.data.shape[1]
    classes = trainset.num_class
    
    # Classifier
    if args.classifier == 'lr':
        classifier = LogisticRegression
    elif args.classifier == 'mlp':
        classifier = MLP
    else:
        print('Error: Unkown classifier')
        exit(1)
    
    # Dynamic predictor
    DP = DynamicPredictor(in_size=classes, d_model=classes*2).to(device)
    d_optimizer = optim.AdamW(DP.parameters(), lr=d_lr, weight_decay=d_decay)
    
    
    test_acc = []
    activate = 'finetune'
    model = DynseDP(time_window, max_ensemble_size, max_buffer_size, k, device)
    for t in range(num_batch):
        trainset.set_t(t)
        if t >= 1:
            print(f'Test {t}: Using {activate}')
            softmax_log_history = []
            softmax_log_pred_future = []
            dp_sample_idx = []
            y_preds = []
            y_preds_finetune = []
            y_preds_dp = []
            for i, x in enumerate(trainset.data):
                s = model.trace(x)
                if len(s) > 0:
                    softmax_dataset = SoftmaxOnlineDataset([s], mode='test')
                    softmax_data_loader = Data.DataLoader(softmax_dataset, batch_size=1, 
                                                          shuffle=False, num_workers=num_workers)
                    pred_next_softmax = predict_dynamic(softmax_data_loader, DP, device)
                    softmax_log_pred_future += pred_next_softmax
                    softmax_log_history.append(s)
                    dp_sample_idx.append(i)
                    # Store finetune
                    s_pred = softmax_log_history[-1][-1]
                    y_preds_finetune.append(np.argmax(s_pred))
                    # Store DP
                    s_pred = softmax_log_pred_future[-1]
                    y_preds_dp.append(np.argmax(s_pred))
                    if activate == 'finetune':
                        y_preds.append(y_preds_finetune[-1])
                    else:
                        y_preds.append(y_preds_dp[-1])
                else:
                    y_preds.append(model.predict(x))
            acc = np.sum(y_preds == trainset.target) / len(trainset.target)
            print(f'acc:{acc}')
            test_acc.append(acc)
            print(f'Finetune/DP ratio {t}: {len(dp_sample_idx)/len(y_preds)}')
            print(f'Dynse ratio {t}: {(len(y_preds)-len(dp_sample_idx))/len(y_preds)}')
            
        print(f'Train {t}')
        tset, vset = split_train_valid(trainset, train_ratio=train_ratio)
        t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers)
        v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                        shuffle=False, num_workers=num_workers)
                        
        F = classifier(in_size=dim, out_size=classes).to(device)
        optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)            
        
        best_acc = 0
        best_F = None
        p = patience
        for i in range(epochs):
            print('Epoch:', i+1)
            train(t_data_loader, F, optimizer, device)
            loss, acc = test(t_data_loader, F, device)
            print(f'Train loss:{loss}, acc:{acc}')
            loss, acc = test(v_data_loader, F, device)
            print(f'Valid loss:{loss}, acc:{acc}')
            if acc > best_acc:
                best_acc = acc 
                best_F = copy.deepcopy(F)
                p = patience
            else:
                p -= 1
                if p < 0:
                    print('Early stopping!')
                    break
        F = best_F
        
        if t >= 1:    
            print(f'Get data softmax {t}')
            data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                          shuffle=False, num_workers=num_workers)
            _, _, log = test(data_loader, F, device, return_softmax=True)
            for i, j in enumerate(dp_sample_idx):
                softmax_log_history[i].append(log[j])
                
            print(f'Train dynamic predictor {t}')
            softmax_dataset = SoftmaxOnlineDataset(softmax_log_history, mode='train')
            tset, vset = split_train_valid(softmax_dataset, train_ratio=train_ratio)
            t_softmax_data_loader = Data.DataLoader(tset, batch_size=1, 
                                                    shuffle=True, num_workers=num_workers)
            v_softmax_data_loader = Data.DataLoader(vset, batch_size=1, 
                                                    shuffle=False, num_workers=num_workers)
            best_loss = float('inf')
            best_DP = None
            best_opt_state_dict = None 
            p = patience
            for i in range(d_epochs):
                print('Epoch:', i+1)
                train_dynamic(t_softmax_data_loader, DP, d_optimizer, device)
                loss = test_dynamic(t_softmax_data_loader, DP, device)
                print(f'Train loss:{loss}')
                loss = test_dynamic(v_softmax_data_loader, DP, device)
                print(f'Valid loss:{loss}')
                if loss < best_loss:
                    best_loss = loss
                    best_DP = copy.deepcopy(DP)
                    best_opt_state_dict = copy.deepcopy(d_optimizer.state_dict())
                    p = patience
                else:
                    p -= 1
                    if p < 0:
                        print('Early stopping!')
                        break
            DP = best_DP
            d_optimizer = optim.AdamW(DP.parameters(), lr=d_lr, weight_decay=d_decay)
            d_optimizer.load_state_dict(best_opt_state_dict)
                        
        print(f'Add classifier {t}')
        model.add_classifier(F)
        print(f'Add buffer {t}')
        sb = SubspaceBuffer(max_centroids, max_instances)
        sb.add(trainset.data, trainset.target, np.ones(len(trainset.data)))
        model.add_buffer(sb)
        
        if t >= 1:
            acc_finetune = np.sum(np.array(y_preds_finetune) == trainset.target[np.array(dp_sample_idx)]) / len(y_preds_finetune)
            acc_dp = np.sum(np.array(y_preds_dp) == trainset.target[np.array(dp_sample_idx)]) / len(y_preds_dp)
            print(f'Finetune acc {t}: {acc_finetune}')
            print(f'DP acc {t}: {acc_dp}')
            if acc_finetune > acc_dp:
                activate = 'finetune'
            else:
                activate = 'dp'
    
    print('Test acc log:', test_acc)
    print('Mean acc:', sum(test_acc)/len(test_acc))

        
        
    
