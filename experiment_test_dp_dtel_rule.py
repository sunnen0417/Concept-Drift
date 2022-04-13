import numpy as np
import copy
import os
import random
import torch
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from datasets import SoftmaxDataset, BufferDataset
from datasets import dataset_dict
from utils import train, test, train_soft, test_soft, train_dynamic, test_dynamic, predict_dynamic, draw_decision_boundary, split_train_valid, test_dp_dtel_test_ensemble, test_dp_dtel_get_feedback_acc
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
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--lr', type=float, default=2e-3)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-dc', '--decay', type=float, default=5e-5)
    parser.add_argument('-dlr', '--d_lr', type=float, default=1e-3)
    parser.add_argument('-de', '--d_epochs', type=int, default=50)
    parser.add_argument('-ddc', '--d_decay', type=float, default=0)
    parser.add_argument('-fe', '--finetuned_epochs', type=int, default=5)
    parser.add_argument('-adt', '--activate_dynamic_t', type=int, default=3)
    parser.add_argument('-mes', '--max_ensemble_size', type=int, default=25)
    parser.add_argument('-d', '--dataset', type=str, default='translate') # see dataset_dict in datasets.py
    parser.add_argument('-c', '--classifier', type=str, default='lr') # 'lr':logistic regreesion, 'mlp':neural network
    parser.add_argument('-dv', '--device', type=str, default='cuda:0')
    parser.add_argument('-tr', '--train_ratio', type=float, default=0.8)
    parser.add_argument('-p', '--patience', type=int, default=7)
    parser.add_argument('-ltc', '--life_time_coefficient', type=float, default=1.0)
    parser.add_argument('-a', '--alpha', type=float, default=0.5)
    parser.add_argument('-v', '--voting', type=str, default='soft')
    parser.add_argument('-moc', '--mask_old_classifier', action="store_true")
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
    finetuned_epochs = args.finetuned_epochs
    activate_dynamic_t = args.activate_dynamic_t
    max_ensemble_size = args.max_ensemble_size
    train_ratio = args.train_ratio
    patience = args.patience
    life_time_coefficient = args.life_time_coefficient
    alpha = args.alpha
    voting = args.voting
    mask_old_classifier = args.mask_old_classifier
    
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
    classifier_list = []
    w = []
    pred_classifier_list = []
    pred_w = []
    for t in range(num_batch):
        trainset.set_t(t)
        data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)
        if t > 0:
            print(f'Test {t}')
            softmax_log = []
            for F in classifier_list:
                _, _, log = test(data_loader, F, device, return_softmax=True)
                softmax_log.append(log)
                F.cpu()
            softmax_dataset = SoftmaxDataset(softmax_log, mode='test')
            softmax_data_loader = Data.DataLoader(softmax_dataset, batch_size=batch_size, 
                                                  shuffle=False, num_workers=num_workers)
            pred_next_softmax = predict_dynamic(softmax_data_loader, DP, device)
            
            data_loader = Data.DataLoader(trainset, batch_size=1, 
                                          shuffle=False, num_workers=num_workers)
            if t > activate_dynamic_t:
                soft_dataset = BufferDataset(trainset.data, pred_next_softmax, target_type='soft')
                tset, vset = split_train_valid(soft_dataset, train_ratio=train_ratio)
                t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                                shuffle=True, num_workers=num_workers)
                v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                                shuffle=False, num_workers=num_workers)
                
                for F_ in classifier_list:
                    F = copy.deepcopy(F_)
                    F.to(device)
                    optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)
                
                    best_loss = float('inf')
                    best_F = None
                    best_opt_state_dict = None 
                    p = patience
                    for i in range(finetuned_epochs):
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
                    pred_classifier_list.append(F)
                if mask_old_classifier:
                    acc = test_dp_dtel_test_ensemble(data_loader, [], [], pred_classifier_list, pred_w, classes, device, voting=voting)
                else:
                    acc = test_dp_dtel_test_ensemble(data_loader, classifier_list, w, pred_classifier_list, pred_w, classes, device, voting=voting)
            else:
                _, acc = test(data_loader, classifier_list[-1], device)
                
            print(f'acc:{acc}')
            test_acc.append(acc)
            
        print(f'Train {t}')
        tset, vset = split_train_valid(trainset, train_ratio=train_ratio)
        t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers)
        v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                        shuffle=False, num_workers=num_workers)
        
        if len(classifier_list) == 0:                
            F = classifier(in_size=dim, out_size=classes).to(device)
        else:
            F = copy.deepcopy(classifier_list[-1]).to(device)
        optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)            
        
        best_acc = -1
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

        if t > 0:
            if t == num_batch - 1:
                break
        
        if t > 0:    
            print(f'Get data softmax {t}')
            data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                          shuffle=False, num_workers=num_workers)
            _, _, log = test(data_loader, F, device, return_softmax=True)
            softmax_log.append(log)
                
            print(f'Train dynamic predictor {t}')
            softmax_dataset = SoftmaxDataset(softmax_log, mode='train')
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
        
        
        print(f'Update weight {t}')
        data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                      shuffle=False, num_workers=num_workers)
        if len(classifier_list) == 0:
            loss, acc = test(v_data_loader, F, device)
            w.append(acc)
        else:
            feedback, pred_feedback = test_dp_dtel_get_feedback_acc(data_loader, classifier_list, pred_classifier_list, device)
            w = alpha * np.array(feedback) + (1 - alpha) * np.array(w)
            pred_w = alpha * np.array(pred_feedback) + (1 - alpha) * np.array(pred_w)
            w = w.tolist()
            pred_w = pred_w.tolist()
            w.insert(0, w[0] * life_time_coefficient)
            
        print(f'Add classifier {t}')
        classifier_list.append(F)
        if len(classifier_list) > max_ensemble_size:
            classifier_list.pop(0)
            w.pop(0)
        if t == activate_dynamic_t:
            pred_w = [w[-1]] * len(w)
        elif t > activate_dynamic_t:
            pred_w = [np.mean(pred_w)] * len(w)

        pred_classifier_list = []
        print(f'weight:{w}, {pred_w}')
        
    print('Test acc log:', test_acc)
    test_acc = np.array(test_acc)
    print(f'Mean acc ± std: {np.mean(test_acc)} ± {np.std(test_acc)}')

        
        
    
