import numpy as np
import copy
import os
import random
import torch
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from datasets import SoftmaxDataset, BufferDataset, StoreDataset
from datasets import dataset_dict
from utils import train, test, train_soft, test_soft, train_dynamic, test_dynamic, predict_dynamic, train_cost_sensitive, test_cost_sensitive, draw_decision_boundary, split_train_valid
from models import LogisticRegression, MLP, DynamicPredictor
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
    parser.add_argument('-lsm', '--last_step_method', type=str, default='none') # 'none', 'hard', 'soft', 'cost'
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--lr', type=float, default=2e-3)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-dc', '--decay', type=float, default=5e-5)
    parser.add_argument('-dlr', '--d_lr', type=float, default=1e-3)
    parser.add_argument('-de', '--d_epochs', type=int, default=50)
    parser.add_argument('-ddc', '--d_decay', type=float, default=0)
    parser.add_argument('-adt', '--activate_dynamic_t', type=int, default=3)
    parser.add_argument('-tw', '--time_window', type=int, default=3)
    parser.add_argument('-d', '--dataset', type=str, default='translate') # see dataset_dict in datasets.py
    parser.add_argument('-c', '--classifier', type=str, default='lr') # 'lr':logistic regreesion, 'mlp':neural network
    parser.add_argument('-ckpt', '--ckpt_dir', type=str, default='./ckpt_all')
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
    last_step_method = args.last_step_method
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
    activate_dynamic_t = args.activate_dynamic_t
    time_window = args.time_window
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    train_ratio = args.train_ratio
    patience = args.patience
    draw_boundary = False
    
    # Dataset
    trainset = dataset_dict[args.dataset]
    storeset = StoreDataset()
    num_batch = trainset.num_batch
    dim = trainset.data.shape[1]
    classes = trainset.num_class
    
    # Visualize the decision boundary if the dataset is suitable 
    if dim == 2 and classes == 2:
        draw_boundary = True
        img_dir = f'{args.dataset}_cls_all_dp_all_{last_step_method}'
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
    
    # Dynamic predictor
    DP = DynamicPredictor(in_size=classes, d_model=classes*2).to(device)
    d_optimizer = optim.AdamW(DP.parameters(), lr=d_lr, weight_decay=d_decay)
    
    # time_window * n_sample * dim
    data_softmax_history = []
    test_acc = []
    for t in range(num_batch):
        trainset.set_t(t)

        # store batch t dataset
        storeset.append(trainset.data, trainset.target)

        if t > 0:
            print(f'Test {t}')
            data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                    shuffle=False, num_workers=num_workers)
            loss, acc = test(data_loader, F, device)
            print(f'loss:{loss}, acc:{acc}')
            test_acc.append(acc)
        
        print(f'Train {t}')
        tset, vset = split_train_valid(storeset, train_ratio=train_ratio)
        t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers)
        v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                    shuffle=False, num_workers=num_workers)
        best_acc = 0
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
        torch.save(F.state_dict(), f'{ckpt_dir}/{t}.pth')
        
        if t > 0:
            if draw_boundary:
                draw_decision_boundary(data_loader, F, device, x_range=trainset.x_range, y_range=trainset.y_range, newfig=False, db_color='g')
                plt.savefig(f'{img_dir}/batch{t-1}.png')
            if t == num_batch - 1:
                break
        
        if draw_boundary:
            draw_decision_boundary(data_loader, F, device, x_range=trainset.x_range, y_range=trainset.y_range, newfig=True, db_color='b')
            
        print(f'Get data softmax {t}')
        if t > 0:
            data, target = torch.FloatTensor(trainset.data), torch.LongTensor(trainset.target)
            historyset = Data.TensorDataset(data, target)
            data_loader = Data.DataLoader(historyset)
            for i in range(len(data_softmax_history)):
                idx = i + max(0, t-time_window)
                print(f'Load history classifier {idx}')
                H.load_state_dict(torch.load(f'{ckpt_dir}/{idx}.pth'))
                _, _, log = test(data_loader, H, device, return_softmax=True)
                data_softmax_history[i] += log

        data_loader = Data.DataLoader(storeset, batch_size=batch_size, 
                                      shuffle=False, num_workers=num_workers)
        _, _, log = test(data_loader, F, device, return_softmax=True)
        data_softmax_history.append(log)
        
        if t > 0:
            print(f'Train dynamic predictor {t}')
            softmax_dataset = SoftmaxDataset(data_softmax_history, mode='train')
            tset, vset = split_train_valid(softmax_dataset, train_ratio=train_ratio)
            t_softmax_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=num_workers)
            v_softmax_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
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
        
        if len(data_softmax_history) > time_window:
            data_softmax_history.pop(0)
                        
        if last_step_method != 'none':            
            if t >= activate_dynamic_t:
                print(f'Predict data softmax {t+1}')
                softmax_dataset = SoftmaxDataset(data_softmax_history, mode='test')
                softmax_data_loader = Data.DataLoader(softmax_dataset, batch_size=batch_size, 
                                                      shuffle=False, num_workers=num_workers)
                pred_next_softmax = predict_dynamic(softmax_data_loader, DP, device)
        
                print(f'Predict decision boundary {t+1}')
                if last_step_method == 'soft':
                    soft_storeset = BufferDataset(storeset.data, pred_next_softmax, target_type='soft')
                    tset, vset = split_train_valid(soft_storeset, train_ratio=train_ratio)
                    t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=num_workers)
                    v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                                  shuffle=False, num_workers=num_workers)
                    best_loss = float('inf')
                    best_F = None
                    best_opt_state_dict = None 
                    p = patience
                    for i in range(epochs):
                        print('Epoch:', i+1)
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
                    F = best_F
                    optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)
                    optimizer.load_state_dict(best_opt_state_dict)
                elif last_step_method == 'hard':
                    y = [np.argmax(s) for s in np.array(pred_next_softmax)] 
                    y = np.array(y, dtype='int64')
                    storeset.target = y
                    tset, vset = split_train_valid(storeset, train_ratio=train_ratio)
                    t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=num_workers)
                    v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                                  shuffle=False, num_workers=num_workers)
                    best_acc = 0
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
                elif last_step_method == 'cost':
                    soft_storeset = BufferDataset(storeset.data, pred_next_softmax, target_type='soft')
                    tset, vset = split_train_valid(soft_storeset, train_ratio=train_ratio)
                    t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=num_workers)
                    v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                                  shuffle=False, num_workers=num_workers)
                    best_acc = 0
                    best_F = None
                    best_opt_state_dict = None 
                    p = patience
                    for i in range(epochs):
                        print('Epoch:', i+1)
                        train_cost_sensitive(t_data_loader, F, optimizer, device)
                        loss, acc = test_cost_sensitive(t_data_loader, F, device)
                        print(f'Train loss:{loss}, acc:{acc}')               
                        loss, acc = test_cost_sensitive(v_data_loader, F, device)
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
        
        if draw_boundary:      
            data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers)        
            draw_decision_boundary(data_loader, F, device, x_range=trainset.x_range, y_range=trainset.y_range, newfig=False, db_color='r')
    
    print('Test acc log:', test_acc)
    print('Mean acc:', sum(test_acc)/len(test_acc))

        
        
    
