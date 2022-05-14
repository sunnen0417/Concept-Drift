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
from utils import dtel_test_ensemble, refine_ca, dtel_test_ensemble_ca, train, test, train_soft, test_soft, train_dynamic, test_dynamic, predict_dynamic, train_cost_sensitive, test_cost_sensitive, draw_decision_boundary, split_train_valid
from models import LogisticRegression, MLP, DynamicPredictor
import argparse
import warnings
warnings.filterwarnings("ignore")

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
    parser.add_argument('-lsm', '--last_step_method', type=str, default='soft')
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--lr', type=float, default=2e-3)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-dc', '--decay', type=float, default=5e-5)
    parser.add_argument('-dlr', '--d_lr', type=float, default=1e-3)
    parser.add_argument('-de', '--d_epochs', type=int, default=50)
    parser.add_argument('-ddc', '--d_decay', type=float, default=0)
    parser.add_argument('-adt', '--activate_dynamic_t', type=int, default=3)
    parser.add_argument('-d', '--dataset', type=str, default='translate')
    parser.add_argument('-c', '--classifier', type=str, default='lr')
    parser.add_argument('-dv', '--device', type=str, default='cuda:0')
    parser.add_argument('-tr', '--train_ratio', type=float, default=0.8)
    parser.add_argument('-p', '--patience', type=int, default=7)
    ##### dynse
    parser.add_argument('-D', '--max_pool_size', type=int, default=3)
    parser.add_argument('-M', '--max_validation_window_size', type=int, default=3)
    parser.add_argument('-K', '--neighbor_size', type=int, default=5)
    ##### ensemble finetune
    parser.add_argument('-fe', '--finetuned_epochs', type=int, default=5)
    ##### cluster assumption 
    parser.add_argument("-ca", "--cluster_assumption", action="store_true")
    parser.add_argument('-cae', '--ca_epochs', type=int, default=10)
    parser.add_argument('-calr', '--ca_lr', type=float, default=0.0005)
    parser.add_argument('-ema', '--ema_decay', type=float, default=0.998)
    parser.add_argument('-pr', '--perturb_radius', type=float, default=0.1)
    parser.add_argument('-xi', '--XI', type=float, default=0.1)
    #####
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
    #time_window = args.time_window
    train_ratio = args.train_ratio
    patience = args.patience
    #####
    max_pool_size = args.max_pool_size  # D
    max_validation_window_size = args.max_validation_window_size   # M
    neighbor_size = args.neighbor_size  # K
    classifier_pool = []
    pred_classifier_pool = []
    storeset = StoreDataset()  # validation set Q
    finetuned_epochs = args.finetuned_epochs
    cluster_assumption = args.cluster_assumption
    ca_epochs = args.ca_epochs
    ca_lr = args.ca_lr
    ema_decay = args.ema_decay
    perturb_radius = args.perturb_radius
    XI = args.XI
    #####  
    
    # Dataset
    trainset = dataset_dict[args.dataset]
    #storeset = StoreDataset()
    num_batch = trainset.num_batch
    dim = trainset.data.shape[1]
    classes = trainset.num_class
    
    
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
        data_loader = Data.DataLoader(trainset, batch_size=1, 
                        shuffle=False, num_workers=num_workers)

        if t > 0:
            print(f'Test {t}')
            if not cluster_assumption: # ensemble (soft vote)     
                loss, acc = dtel_test_ensemble(data_loader, classifier_pool, pred_classifier_pool, device)        
            else:  # ensemble (cluster assumption weighted vote)  
                data_loader_ca = Data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                F_ca = copy.deepcopy(F)
                refine_ca(ca_epochs, ca_lr, ema_decay, XI, perturb_radius, data_loader_ca, F_ca, device) 
                loss, acc = dtel_test_ensemble_ca(data_loader, storeset, F_ca, classifier_pool, pred_classifier_pool, neighbor_size, device)                                       
            print(f'loss:{loss}, acc:{acc}')
            test_acc.append(acc)
        
        print(f'Train {t}')
        tset, vset = split_train_valid(trainset, train_ratio=train_ratio)
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
        
        # Update E
        classifier_pool.append(copy.deepcopy(F))
        if len(classifier_pool) > max_pool_size:
            classifier_pool.pop(0)
        # Update E'
        pred_classifier_pool = []
        # Update storeset
        storeset.append(trainset.data, trainset.target)  
        if len(storeset.batch_data_count) > max_validation_window_size:
            storeset.remove_oldest_batch()
        
        if t > 0:
            if t == num_batch - 1:
                break
        
            
        print(f'Get data softmax {t}')
        dataset = BufferDataset(storeset.data, storeset.target)
        data_loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        data_softmax_history = []   
        for F_previous in classifier_pool[max(-max_validation_window_size, -len(classifier_pool)):]: 
            _, _, log = test(data_loader, F_previous, device, return_softmax=True)  
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
        
        if len(data_softmax_history) >= max_validation_window_size:
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
                    
                    for cls_ in classifier_pool:
                        cls = copy.deepcopy(cls_)
                        cls_optimizer = optim.Adam(cls.parameters(), lr=lr, weight_decay=decay)
                        best_loss = float('inf')
                        best_F = None
                        p = patience
                        
                        for i in range(finetuned_epochs):
                            print('Step3 Epoch:', i+1) 
                            train_soft(t_data_loader, cls, cls_optimizer, device)
                            loss = test_soft(t_data_loader, cls, device)
                            print(f'Train loss:{loss}')
                            loss = test_soft(v_data_loader, cls, device)
                            print(f'Valid loss:{loss}')         
                            if loss < best_loss:
                                best_loss = loss
                                best_F = copy.deepcopy(cls)
                                p = patience
                            else:
                                p -= 1
                                if p < 0:
                                    print('Early stopping!')
                                    break
                        F_ = best_F            
                        pred_classifier_pool.append(copy.deepcopy(F_))              

                elif last_step_method == 'hard':
                    y = [np.argmax(s) for s in np.array(pred_next_softmax)] 
                    y = np.array(y, dtype='int64')
                    hard_storeset = BufferDataset(storeset.data, y, target_type='hard')
                    tset, vset = split_train_valid(hard_storeset, train_ratio=train_ratio)
                    t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=num_workers)
                    v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                                  shuffle=False, num_workers=num_workers)
                    
                    for cls_ in classifier_pool:
                        cls = copy.deepcopy(cls_)
                        cls_optimizer = optim.Adam(cls.parameters(), lr=lr, weight_decay=decay)
                        cls.train()
                        best_acc = 0
                        best_F = None
                        p = patience
                        for i in range(finetuned_epochs):
                            print('Step3 Epoch:', i+1)
                            train(t_data_loader, cls, cls_optimizer, device)
                            loss, acc = test(t_data_loader, cls, device)
                            print(f'Train loss:{loss}, acc:{acc}')
                            loss, acc = test(v_data_loader, cls, device)
                            print(f'Valid loss:{loss}, acc:{acc}')               
                            if acc > best_acc:
                                best_acc = acc
                                best_F = copy.deepcopy(cls)
                                p = patience
                            else:
                                p -= 1
                                if p < 0:
                                    print('Early stopping!')
                                    break
                                    
                        pred_classifier_pool.append(copy.deepcopy(best_F))   
                        
                elif last_step_method == 'cost':
                    soft_storeset = BufferDataset(storeset.data, pred_next_softmax, target_type='soft')
                    tset, vset = split_train_valid(soft_storeset, train_ratio=train_ratio)
                    t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=num_workers)
                    v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                                  shuffle=False, num_workers=num_workers)
                    
                    for cls_ in classifier_pool:
                        cls = copy.deepcopy(cls_)
                        cls_optimizer = optim.Adam(cls.parameters(), lr=lr, weight_decay=decay)
                        cls.train()
                        best_acc = 0
                        best_F = None
                        p = patience
                        for i in range(finetuned_epochs):
                            print('Step3 Epoch:', i+1)
                            train_cost_sensitive(t_data_loader, cls, cls_optimizer, device)
                            loss, acc = test_cost_sensitive(t_data_loader, cls, device)
                            print(f'Train loss:{loss}, acc:{acc}')               
                            loss, acc = test_cost_sensitive(v_data_loader, cls, device)
                            print(f'Valid loss:{loss}, acc:{acc}')               
                            if acc > best_acc:
                                best_acc = acc
                                best_F = copy.deepcopy(cls)
                                p = patience
                            else:
                                p -= 1
                                if p < 0:
                                    print('Early stopping!')
                                    break
                                    
                        pred_classifier_pool.append(copy.deepcopy(best_F))          
    
    print('Test acc log:', test_acc)
    test_acc = np.array(test_acc)
    print(f'Mean acc ± std: {np.mean(test_acc)} ± {np.std(test_acc)}')