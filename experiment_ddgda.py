import numpy as np
import copy
import os
import random
import torch
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from datasets import BufferDataset, StoreDataset
from datasets import dataset_dict
from utils import train, test, train_soft, test_soft, draw_decision_boundary, split_train_valid, train_ddgda, test_ddgda
from models import LogisticRegression, MLP
import argparse
import math

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
    parser.add_argument('-qlr', '--Q_lr', type=float, default=1e-3)
    parser.add_argument('-qe', '--Q_epochs', type=int, default=50)
    parser.add_argument('-qdc', '--Q_decay', type=float, default=0)
    parser.add_argument('-adt', '--activate_dynamic_t', type=int, default=3)
    parser.add_argument('-tw', '--time_window', type=int, default=3)
    parser.add_argument('-d', '--dataset', type=str, default='translate') # see dataset_dict in datasets.py  
    parser.add_argument('-c', '--classifier', type=str, default='lr') # 'lr':logistic regreesion, 'mlp':neural network
    parser.add_argument('-dv', '--device', type=str, default='cuda:0')
    parser.add_argument('-tr', '--train_ratio', type=float, default=0.8)
    parser.add_argument('-p', '--patience', type=int, default=7)
    parser.add_argument('-fcm', '--finetune_classifier_method', type=str, default='soft')
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
    Q_lr = args.Q_lr
    Q_epochs = args.Q_epochs
    Q_decay = args.Q_decay
    activate_dynamic_t = args.activate_dynamic_t
    time_window = args.time_window
    draw_boundary = False
    finetune_classifier_method = args.finetune_classifier_method
    
    # Get minimum chunk size
    dataset = dataset_dict[args.dataset]
    min_chunk_size = math.inf
    for t in range(dataset.num_batch):
        dataset.set_t(t)
        min_chunk_size = min(min_chunk_size, len(dataset.data))
    print('min_chunk_size', min_chunk_size)
    
    # Dataset
    trainset = dataset_dict[args.dataset]
    storeset = StoreDataset()
    num_batch = trainset.num_batch
    dim = trainset.data.shape[1]
    classes = trainset.num_class
    train_ratio = args.train_ratio
    patience = args.patience
    
    # Visualize the decision boundary if the dataset is suitable 
    if dim == 2:
        draw_boundary = True
        img_dir = f'{args.dataset}_num_batch_{trainset.num_batch}_ddgda_seed_{seed}'
        os.makedirs(img_dir, exist_ok=True)
    
    # Classifier
    if args.classifier == 'lr':
        F = LogisticRegression(in_size=dim, out_size=classes).to(device)
        H = LogisticRegression(in_size=dim, out_size=classes).to(device)
    elif args.classifier == 'mlp':
        F = MLP(in_size=dim, out_size=classes).to(device)
        H = MLP(in_size=dim, out_size=classes).to(device)
    else:
        print('Error: Unkown classifier')
        exit(1)
    optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)
    
    # Q (resampler)
    Q = torch.randn(min_chunk_size)*math.sqrt(1/min_chunk_size)  # Xavier Initialization
    Q = Q.to(device)
    Q.requires_grad = True
    Q_optimizer = optim.AdamW([Q], lr=Q_lr, weight_decay=Q_decay)

    # time_window * n_sample * dim
    data_history = []
    test_acc = []
    for t in range(num_batch):
        trainset.set_t(t)
        data_loader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
        
        # Re-train classifier
        if t > activate_dynamic_t:  
            # generate predicted dataset
            loss, acc, pred_soft_label, pred_hard_label = test_ddgda(min_chunk_size, data_history[-1], trainset, Q, device)
            print(f'Test proxy model, loss:{loss}, acc:{acc}')
            if finetune_classifier_method == 'soft':
                storeset = BufferDataset(trainset.data, pred_soft_label.cpu(), target_type='soft')
            else:
                storeset = BufferDataset(trainset.data, pred_hard_label.cpu(), target_type='hard')
            
            # re-train classifier
            print(f'Re-train classifier {t}')
            tset, vset = split_train_valid(storeset, train_ratio=train_ratio)
            t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers)
            v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                        shuffle=False, num_workers=num_workers)
            best_acc = -1
            best_loss = float('inf')
            best_F = None
            best_opt_state_dict = None 
            p = patience
            for i in range(epochs):
                print('Epoch:', i+1)
                if finetune_classifier_method == 'soft':
                    ### soft label training
                    train_soft(t_data_loader, F, optimizer, device)
                    loss = test_soft(t_data_loader, F, device)
                    print(f'Train loss:{loss}')
                    loss = test_soft(v_data_loader, F, device)
                    print(f'Valid loss:{loss}')
                    if loss < best_loss:
                        best_loss = loss
                        best_F = copy.deepcopy(F)
                        best_opt_state_dict = copy.deepcopy(optimizer.state_dict())
                        p = patience
                    else:
                        p -= 1
                        if p < 0:
                            print('Early stopping!')
                            break     
                else:
                    ### hard label training     
                    train(t_data_loader, F, optimizer, device)
                    print(f'Train loss:{loss}, acc:{acc}')
                    loss, acc = test(v_data_loader, F, device)     
                    print(f'Valid loss:{loss}, acc:{acc}')
                    if acc > best_acc:
                        best_acc = acc 
                        best_F = copy.deepcopy(F)
                        best_opt_state_dict = copy.deepcopy(optimizer.state_dict())
                        p = patience
                    else:
                        p -= 1
                        if p < 0:
                            print('Early stopping!')
                            break
                        
            F = best_F
            optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)
            optimizer.load_state_dict(best_opt_state_dict)
       
        if t > 0:
            if draw_boundary:      
                data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                            shuffle=True, num_workers=num_workers)        
                draw_decision_boundary(data_loader, F, device, classes, x_range=trainset.x_range, y_range=trainset.y_range, newfig=False, db_color='r')
                
        
        print(f'Test {t}')
        if t > 0:
            loss, acc = test(data_loader, F, device)
            print(f'loss:{loss}, acc:{acc}')
            test_acc.append(acc)
            
        print(f'Train {t}')
        tset, vset = split_train_valid(trainset, train_ratio=train_ratio)
        t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers)
        v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                    shuffle=False, num_workers=num_workers)
        best_acc = -1
        best_F = None
        best_opt_state_dict = None 
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
                best_opt_state_dict = copy.deepcopy(optimizer.state_dict())
                p = patience
            else:
                p -= 1
                if p < 0:
                    print('Early stopping!')
                    break
        F = best_F
        optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)
        optimizer.load_state_dict(best_opt_state_dict)
        
        if t > 0:
            if draw_boundary:
                draw_decision_boundary(data_loader, F, device, classes, x_range=trainset.x_range, y_range=trainset.y_range, newfig=False, db_color='g')
                plt.savefig(f'{img_dir}/batch{t-1}.png', bbox_inches='tight')
            if t == num_batch - 1:
                break
        
        if draw_boundary:
            draw_decision_boundary(data_loader, F, device, classes, x_range=trainset.x_range, y_range=trainset.y_range, newfig=True, db_color='b')
            
            
        data_history.append(copy.deepcopy(trainset))
        if len(data_history) > time_window+1: # time_window=3 -> store 4 batches
            data_history.pop(0)
             
        print(f'Train Q {t}')
        if t > 0:
            for i in range(Q_epochs):
                print('Epoch:', i+1)
                train_ddgda(min_chunk_size, data_history, Q, Q_optimizer, device)


    print('Test acc log:', test_acc)
    test_acc = np.array(test_acc)
    print(f'Mean acc ± std: {np.mean(test_acc)} ± {np.std(test_acc)}')
