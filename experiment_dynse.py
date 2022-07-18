import numpy as np
import copy
import random
import torch
import torch.optim as optim
import torch.utils.data as Data
from datasets import StoreDataset
from datasets import dataset_dict
from utils import train, test, test_dynse, split_train_valid
from models import LogisticRegression, MLP
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
    parser.add_argument('-d', '--dataset', type=str, default='translate') # see dataset_dict in datasets.py
    parser.add_argument('-c', '--classifier', type=str, default='lr') # 'lr':logistic regreesion, 'mlp':neural network
    parser.add_argument('-dv', '--device', type=str, default='cuda:0')
    parser.add_argument('-tr', '--train_ratio', type=float, default=0.8)
    parser.add_argument('-p', '--patience', type=int, default=7)
    parser.add_argument('-mes', '--max_ensemble_size', type=int, default=3)  #D
    parser.add_argument('-mvws', '--max_validation_window_size', type=int, default=1) #M
    parser.add_argument('-k', '--neighbor_size', type=int, default=5) #K
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
    train_ratio = args.train_ratio
    patience = args.patience
    max_ensemble_size = args.max_ensemble_size  # D
    max_validation_window_size = args.max_validation_window_size   # M
    neighbor_size = args.neighbor_size  # K
    classifier_pool = []
    validation_set = StoreDataset()   # validation set Q
    
    # Dataset
    trainset = dataset_dict[args.dataset]
    num_batch = trainset.num_batch
    dim = trainset.data.shape[1]
    classes = trainset.num_class
    
    
    # Classifier
    if args.classifier == 'lr':
        F = LogisticRegression(in_size=dim, out_size=classes).to(device)
    elif args.classifier == 'mlp':
        F = MLP(in_size=dim, out_size=classes).to(device)
    else:
        print('Error: Unkown classifier')
        exit(1)
    optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)
    
    test_acc = []
    for t in range(num_batch):
        trainset.set_t(t)        
        data_loader = Data.DataLoader(trainset, batch_size=1, 
                                shuffle=False, num_workers=num_workers)  # batch_size=1
        if t > 0:
            print(f'Test {t}')
            #loss, acc = test(data_loader, F, device)
            loss, acc = test_dynse(data_loader, neighbor_size, validation_set, classifier_pool, device)
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
        
        classifier_pool.append(copy.deepcopy(F))
        if len(classifier_pool) > max_ensemble_size:
            classifier_pool.pop(0)
        validation_set.append(trainset.data, trainset.target)  
        if len(validation_set.batch_data_count) > max_validation_window_size:
            validation_set.remove_oldest_batch()
        
                        
    print('Test acc log:', test_acc)
    test_acc = np.array(test_acc)
    print(f'Mean acc ± std: {np.mean(test_acc)} ± {np.std(test_acc)}')
