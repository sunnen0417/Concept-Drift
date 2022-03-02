import torch
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from datasets import dataset_dict
from utils import train, test, draw_decision_boundary, split_train_valid
from utils import get_correct_status, Q_statistic, MSE_gamma, MSE_i, test_ensemble, diversity
from models import MLP, LogisticRegression
import copy

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
    parser.add_argument('-d', '--dataset', type=str, default='translate') # see dataset_dict in datasets.py
    parser.add_argument('-c', '--classifier', type=str, default='lr') # 'lr':logistic regreesion, 'mlp':neural network
    parser.add_argument('-dv', '--device', type=str, default='cuda:0')
    parser.add_argument('-tr', '--train_ratio', type=float, default=0.8)
    parser.add_argument('-p', '--patience', type=int, default=7)
    parser.add_argument('-nc', '--num_classifiers', type=int, default=3)
    parser.add_argument('-fe', '--finetuned_epochs', type=int, default=1)
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-5)
    return parser

if __name__ == '__main__':
    ## Get arguments
    parser = get_parser()
    args = parser.parse_args()
    
    ## Hyperparameters
    seed = args.seed
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('using', device)
    num_workers = args.num_workers
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    decay = args.decay
    train_ratio = args.train_ratio
    patience = args.patience
    num_classifiers = args.num_classifiers
    finetuned_epochs = args.finetuned_epochs
    epsilon = args.epsilon
    
    ## Dataset
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
        
    classifier_list = []
    finetuned_classifier_list = []
    weight_list = []
    test_acc = []  
    ## DTEL
    for t in range(num_batch):
        trainset.set_t(t)
        data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)
        
        # Testing
        if t > 0:
            print(f'Test {t}')
            loss, acc = test_ensemble(data_loader, finetuned_classifier_list, weight_list, classes, device)
            print(f'loss:{loss}, acc:{acc}')
            test_acc.append(acc)

        # Training
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
        best_opt_state_dict = None 
        p = patience
        
        # Train new base model  
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

        # transfer historical classifier
        finetuned_classifier_list = []
        data_loader2 = Data.DataLoader(trainset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)
        for i in range(len(classifier_list)):
            finetuned_classifier_list.append(copy.deepcopy(classifier_list[i]))
            optimizer = optim.Adam(finetuned_classifier_list[i].parameters(), lr=lr, weight_decay=decay)
            for j in range(finetuned_epochs):
                train(data_loader2, finetuned_classifier_list[i], optimizer, device)
        finetuned_classifier_list.append(copy.deepcopy(best_F))

        # select classifiers
        if len(classifier_list) < num_classifiers:
            classifier_list.append(best_F)
        else:
            classifier_list.append(best_F)
            Q_map = [[0 for j in range(len(classifier_list))] for i in range(len(classifier_list))]

            correct_status_list = []
            for i in range(len(classifier_list)):
                correct_status = get_correct_status(classifier_list[i], data_loader, device)
                correct_status_list.append(correct_status)
                
            for i in range(len(classifier_list)):
                for j in range(i + 1, len(classifier_list)):
                    q = Q_statistic(correct_status_list[i], correct_status_list[j], epsilon)
                    Q_map[i][j] = q
            
            max_div = -9999999999999
            toRemove = -1
            for index in range(len(classifier_list)):
                div = diversity(Q_map, index)
                if div > max_div:
                    max_div = div
                    toRemove = index
            classifier_list.pop(toRemove)
            
        # get weights
        weight_list = []
        mse_g = MSE_gamma(data_loader, classes, device)
        for cls in finetuned_classifier_list[:-1]:
            mse_i = MSE_i(cls, data_loader, device)
            weight = 1 / (mse_g + mse_i + epsilon)
            weight_list.append(weight)
        weight = 1 / (mse_g + epsilon)
        weight_list.append(weight)

    print('Test acc log:', test_acc)
    test_acc = np.array(test_acc)
    print(f'Mean acc ± std: {np.mean(test_acc)} ± {np.std(test_acc)}')