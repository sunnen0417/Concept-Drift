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
from utils import train, test, train_soft, test_soft, train_dynamic, test_dynamic, predict_dynamic, train_cost_sensitive, test_cost_sensitive, draw_decision_boundary, sample, update_all_vaes
from models import LogisticRegression, MLP, DynamicPredictor, VAE
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
    parser.add_argument('-vlr', '--vae_lr', type=float, default=1e-3)
    parser.add_argument('-ve', '--vae_epochs', type=int, default=250)
    parser.add_argument('-vdc', '--vae_decay', type=float, default=0)
    parser.add_argument('-th', '--theta', type=float, default=0.1)
    parser.add_argument('-sn', '--sample_n', type=int, default=200)
    parser.add_argument('-eps', '--eps', type=float, default=0.1)
    parser.add_argument('-adt', '--activate_dynamic_t', type=int, default=3)
    parser.add_argument('-tw', '--time_window', type=int, default=3)
    parser.add_argument('-d', '--dataset', type=str, default='translate') # see dataset_dict in datasets.py
    parser.add_argument('-c', '--classifier', type=str, default='lr') # 'lr':logistic regreesion, 'mlp':neural network
    parser.add_argument('-dv', '--device', type=str, default='cuda:0')
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
    vae_lr = args.vae_lr
    vae_epochs = args.vae_epochs
    vae_decay = args.vae_decay
    theta = args.theta
    sample_n = args.sample_n
    eps = args.eps
    activate_dynamic_t = args.activate_dynamic_t
    time_window = args.time_window
    draw_boundary = False

    # Dataset
    trainset = dataset_dict[args.dataset]
    num_batch = trainset.num_batch
    dim = trainset.data.shape[1]
    classes = trainset.num_class
    cate_feat = trainset.cate_feat

    # Visualize the decision boundary if the dataset is suitable 
    if dim == 2 and classes == 2:
        draw_boundary = True
        img_dir = f'{args.dataset}_cls_vae_dp_vae_{last_step_method}'
        os.makedirs(img_dir, exist_ok=True)
    
    # Classifier
    if args.classifier == 'lr':
        F = LogisticRegression(in_size=dim, out_size=classes).to(device)
    elif args.classifier == 'mlp':
        F = MLP(in_size=dim, out_size=classes).to(device)
    else:
        print('Error: Unkown classifier')
        exit(1)
    optimizer = optim.Adam(F.parameters(), lr=lr, weight_decay=decay)
        
    # Dynamic predictor
    DP = DynamicPredictor(in_size=classes, d_model=classes*2).to(device)
    d_optimizer = optim.AdamW(DP.parameters(), lr=d_lr, weight_decay=d_decay)
    
    # VAE
    vae_list = []
    vae_optimizer_list = []
    for i in range(classes):
        hidden_size = max(2*dim, 2)
        latent_size = max(int(dim/4), 1)
        vae_list.append(VAE(feat_size=dim, hidden_size=hidden_size, latent_size=latent_size))
        vae_optimizer_list.append(optim.Adam(vae_list[i].parameters(), lr=vae_lr, weight_decay=vae_decay))
    
    classifier_list = []
    test_acc = []
    for t in range(num_batch):
        trainset.set_t(t)
        if t > 0:
            print(f'Test {t}')
            data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                    shuffle=False, num_workers=num_workers)
            loss, acc = test(data_loader, F, device)
            print(f'loss:{loss}, acc:{acc}')
            test_acc.append(acc)
            
        print(f'Update VAEs {t}')
        data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)
        update_all_vaes(data_loader, vae_list, vae_optimizer_list, vae_epochs, theta, sample_n, eps, device)
        
        print(f'Sample data from VAEs {t}')
        aug_data = []
        aug_label = []
        for i in range(classes):
            data = sample(vae_list[i], theta, sample_n, cate_feat, device, batch_size=batch_size, num_workers=num_workers)
            labels = np.full(len(data), i)
            aug_data += data.tolist()
            aug_label += labels.tolist()
        aug_data += trainset.data.tolist()
        aug_label += trainset.target.tolist()
        aug_trainset = BufferDataset(aug_data, aug_label, target_type='hard')
        
        print(f'Train {t}')
        aug_data_loader = Data.DataLoader(aug_trainset, batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers)
        for i in range(epochs):
            print('Epoch:', i+1)
            train(aug_data_loader, F, optimizer, device)
            loss, acc = test(aug_data_loader, F, device)
            print(f'loss:{loss}, acc:{acc}')
        
        if t > 0:
            if draw_boundary:
                draw_decision_boundary(data_loader, F, device, x_range=trainset.x_range, y_range=trainset.y_range, newfig=False, db_color='g')
                plt.savefig(f'{img_dir}/batch{t-1}.png')
            if t == num_batch - 1:
                break
        
        if draw_boundary:
            draw_decision_boundary(data_loader, F, device, x_range=trainset.x_range, y_range=trainset.y_range, newfig=True, db_color='b')         
        
        classifier_list.append(copy.deepcopy(F))
        
        print(f'Get data softmax {t}')
        aug_data_loader = Data.DataLoader(aug_trainset, batch_size=batch_size, 
                                          shuffle=False, num_workers=num_workers)
        # time_window * n_sample * dim
        data_softmax_history = []
        for i in range(len(classifier_list)):
            _, _, log = test(aug_data_loader, classifier_list[i], device, return_softmax=True)
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
        
        if last_step_method != 'none':
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
                
        if draw_boundary:  
            data_loader = Data.DataLoader(trainset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers)        
            draw_decision_boundary(data_loader, F, device, x_range=trainset.x_range, y_range=trainset.y_range, newfig=False, db_color='r')
    
    print('Test acc log:', test_acc)
    print('Mean acc:', sum(test_acc)/len(test_acc))

        
        
    
