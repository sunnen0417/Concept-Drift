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
from utils import train, test, train_soft, test_soft, train_dynamic, test_dynamic, predict_dynamic, draw_decision_boundary, split_train_valid, test_ensemble_3_sets, get_feedback_acc_3_sets, train_ddg_da, test_ddg_da
from models import LogisticRegression, MLP, DynamicPredictor, PredNet
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
    ###
    parser.add_argument('-ddlr', '--dd_lr', type=float, default=1e-3)
    parser.add_argument('-dde', '--dd_epochs', type=int, default=50)
    parser.add_argument('-dddc', '--dd_decay', type=float, default=0)
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
    draw_boundary = False
    ###
    dd_lr = args.dd_lr
    dd_epochs = args.dd_epochs
    dd_decay = args.dd_decay
 
    # Dataset
    trainset = dataset_dict[args.dataset]
    num_batch = trainset.num_batch
    dim = trainset.data.shape[1]
    classes = trainset.num_class
    ###
    chunk_size = trainset.n_per_batch
    
    # Visualize the decision boundary if the dataset is suitable 
    if dim == 2:
        draw_boundary = True
        img_dir = f'{args.dataset}_num_batch_{trainset.num_batch}_3_sets_{seed}'
        os.makedirs(img_dir, exist_ok=True)
        
    # Classifier
    if args.classifier == 'lr':
        classifier = LogisticRegression
    elif args.classifier == 'mlp':
        classifier = MLP
    else:
        print('Error: Unkown classifier')
        exit(1)
    F = classifier(in_size=dim, out_size=classes).to(device)
    optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)

    # Dynamic predictor
    DP = DynamicPredictor(in_size=classes, d_model=classes*2).to(device)
    d_optimizer = optim.AdamW(DP.parameters(), lr=d_lr, weight_decay=d_decay)

    # DDG-DA
    DDG_DA = PredNet(chunk_size).to(device)
    dd_optimizer = optim.AdamW(DDG_DA.parameters(), lr=dd_lr, weight_decay=dd_decay)

    test_acc = []
    classifier_list = []
    w = []
    finetuned_classifier_list = []
    finetuned_w = []
    pred_classifier_list = []
    pred_w = []
    ###
    task_list = []
    for t in range(num_batch):
        trainset.set_t(t)
        data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)
        if t > 0:
            print(f'Test {t}')
            if t > activate_dynamic_t:
                # DDG-DA: training
                print(f'Train DDG-DA {t}')
                dd_loader = []
                for task in range(len(task_list)-1):
                    X = torch.FloatTensor(task_list[task].data)
                    y = torch.FloatTensor(task_list[task].target)
                    X_test = torch.FloatTensor(task_list[task+1].data)
                    target = torch.FloatTensor(task_list[task+1].target)
                    dd_loader.append((X, y, X_test, target))
                for i in range(dd_epochs):
                    # print('Epoch:', i+1)
                    train_ddg_da(dd_loader, DDG_DA, dd_optimizer, device)
                # DDG-DA: testing
                print(f'Test DDG-DA {t}')
                X = torch.FloatTensor(task_list[-1].data)
                y = torch.FloatTensor(task_list[-1].target)
                X_test = torch.FloatTensor(trainset.data)
                target = torch.FloatTensor(trainset.target)
                dd_loader = [(X, y, X_test, target)]
                loss, acc = test_ddg_da(dd_loader, DDG_DA, device)
            else:
                loss, acc = test(data_loader, F, device)

            print(f'acc:{acc}')
            test_acc.append(acc)

        # DDG-DA: store historical data for task_train
        task_list.append(copy.deepcopy(trainset))
        if len(task_list) > max_ensemble_size:
            task_list.pop(0)

        if t < activate_dynamic_t: # the next round of t == activate_dynamic_t will activate DDG-DA
            print(f'Train {t}')
            tset, vset = split_train_valid(trainset, train_ratio=train_ratio)
            t_data_loader = Data.DataLoader(tset, batch_size=batch_size, 
                                            shuffle=True, num_workers=num_workers)
            v_data_loader = Data.DataLoader(vset, batch_size=batch_size, 
                                            shuffle=False, num_workers=num_workers) 
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
        
    print('Test acc log:', test_acc)
    test_acc = np.array(test_acc)
    print(f'Mean acc ± std: {np.mean(test_acc)} ± {np.std(test_acc)}')
