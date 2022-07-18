import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from datasets import BufferDataset, SoftmaxDataset
from sklearn.neighbors import KNeighborsClassifier

# Training
def train(train_loader, F, optimizer, device):
    F.to(device)
    F.train()
    criterion = nn.CrossEntropyLoss()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
       
# Testing
def test(test_loader, F, device, return_softmax=False):
    F.to(device)
    F.eval()
    criterion = nn.CrossEntropyLoss()
    if return_softmax:
        softmax_log = []
    with torch.no_grad():
        correct = 0
        total_loss = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = F(data)
            loss = criterion(output, target)
            total_loss += loss.item() * len(data)
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            if return_softmax:
                prob = nn.functional.softmax(output, dim=1).tolist()
                softmax_log += prob
    acc = correct / len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    if return_softmax:
        return total_loss, acc, softmax_log
    return total_loss, acc

# Training using soft label
def train_soft(train_loader, F, optimizer, device):
    F.to(device)
    F.train()
    criterion = nn.KLDivLoss(reduction='batchmean')
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = nn.functional.log_softmax(F(data), dim=1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Testing using soft label
def test_soft(test_loader, F, device, return_softmax=False):
    F.to(device)
    F.eval()
    criterion = nn.KLDivLoss(reduction='batchmean')
    if return_softmax:
        softmax_log = []
    with torch.no_grad():
        total_loss = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = F(data)
            output = nn.functional.log_softmax(logits, dim=1)
            loss = criterion(output, target)
            total_loss += loss.item() * len(data)
            if return_softmax:
                prob = nn.functional.softmax(logits, dim=1).tolist()
                softmax_log += prob
    total_loss /= len(test_loader.dataset)
    if return_softmax:
        return total_loss, softmax_log
    return total_loss

# cost sensitive loss
class CostSensitiveLoss(nn.Module):
    def __init__(self):
        super(CostSensitiveLoss, self).__init__()
        
    def forward(self, output, target): 
        # output: predict logit; target: ground truth prob
        device = output.device
        output = nn.functional.log_softmax(output, dim=1)
        weight = -torch.sum(target*torch.log2(torch.maximum(target, torch.full(target.shape, 1e-5).to(device))), dim=1)
        class_indices = torch.argmax(target, dim=1)
        data_indices = torch.arange(len(class_indices)).to(device)
        loss = -torch.mean(output[data_indices, class_indices]*weight)
        
        return loss
        
# Training using cost sensitive
def train_cost_sensitive(train_loader, F, optimizer, device):
    F.to(device)
    F.train()
    criterion = CostSensitiveLoss()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Testing using cost sensitive
def test_cost_sensitive(test_loader, F, device, return_softmax=False):
    F.to(device)
    F.eval()
    criterion = CostSensitiveLoss()
    if return_softmax:
        softmax_log = []
    with torch.no_grad():
        correct = 0
        total_loss = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = F(data)
            loss = criterion(output, target)
            total_loss += loss.item() * len(data)
            _, pred = output.max(1)
            _, label = target.max(1)
            correct += pred.eq(label).sum().item()
            if return_softmax:
                prob = nn.functional.softmax(output, dim=1).tolist()
                softmax_log += prob
    acc = correct / len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    if return_softmax:
        return total_loss, acc, softmax_log
    return total_loss, acc

# Training dynamic
def train_dynamic(train_loader, DP, optimizer, device):
    DP.to(device)
    DP.train()
    criterion = nn.KLDivLoss(reduction='batchmean')
    if len(train_loader.dataset[0]) == 2:
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = nn.functional.log_softmax(DP(data), dim=1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    elif len(train_loader.dataset[0]) == 3:
        for data, loc, target in train_loader:
            data, loc, target = data.to(device), loc.to(device), target.to(device)
            optimizer.zero_grad()
            output = nn.functional.log_softmax(DP(data, loc), dim=1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Testing dynamic
def test_dynamic(test_loader, DP, device, return_softmax=False):
    DP.to(device)
    DP.eval()
    criterion = nn.KLDivLoss(reduction='batchmean')
    if return_softmax:
        softmax_log = []
    with torch.no_grad():
        total_loss = 0
        if len(test_loader.dataset[0]) == 2:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = DP(data)
                output = nn.functional.log_softmax(logits, dim=1)
                loss = criterion(output, target)
                total_loss += loss.item() * len(data)
                if return_softmax:
                    prob = nn.functional.softmax(logits, dim=1).tolist()
                    softmax_log += prob
        elif len(test_loader.dataset[0]) == 3:
            for data, loc, target in test_loader:
                data, loc, target = data.to(device), loc.to(device), target.to(device)
                logits = DP(data, loc)
                output = nn.functional.log_softmax(logits, dim=1)
                loss = criterion(output, target)
                total_loss += loss.item() * len(data)
                if return_softmax:
                    prob = nn.functional.softmax(logits, dim=1).tolist()
                    softmax_log += prob
                    
    total_loss /= len(test_loader.dataset)
    if return_softmax:
        return total_loss, softmax_log
    return total_loss

# Predict dynamic (no target)
def predict_dynamic(test_loader, DP, device):
    DP.to(device)
    DP.eval()
    softmax_log = []
    with torch.no_grad():
        if type(test_loader.dataset[0]) is not tuple:
            for data in test_loader:
                data = data.to(device)
                prob = nn.functional.softmax(DP(data), dim=1).tolist()
                softmax_log += prob
        else:
            for data, loc in test_loader:
                data, loc = data.to(device), loc.to(device)
                prob = nn.functional.softmax(DP(data, loc), dim=1).tolist()
                softmax_log += prob            
    return softmax_log

# Only for 2d
def draw_decision_boundary(data_loader, F, device, num_class, x_range=None, y_range=None, newfig=False, db_color='b', dp_colors=['#FF9999', '#99CCFF', '#99FF99', 'c', 'm', 'y', '#F6B26B']):
    if num_class > len(dp_colors):
        print('Error: length of dp_colors should not be lower than num_class')
        return
    
    delta = 0.2
    h = 0.02
    X = []
    y = []
    for data, target in data_loader:
        X += data.tolist()
        y += target.tolist()
    X = np.array(X)
    y = np.array(y)
    if x_range is None:
        x_min = np.amin(np.transpose(X)[0]) - delta 
        x_max = np.amax(np.transpose(X)[0]) + delta
    else:
        x_min = x_range[0] - delta
        x_max = x_range[1] + delta
    if y_range is None:
        y_min = np.amin(np.transpose(X)[1]) - delta
        y_max = np.amax(np.transpose(X)[1]) + delta
    else:
        y_min = y_range[0] - delta
        y_max = y_range[1] + delta
    
    a = np.arange(x_min, x_max, h)
    b = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(a, b)
    x_plot = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x_plot.append([xx[i][j], yy[i][j]])
    x_plot = torch.FloatTensor(x_plot)
    x_plot_dataset = Data.TensorDataset(x_plot)
    x_plot_loader = Data.DataLoader(x_plot_dataset, batch_size=512,
                                    shuffle=False, num_workers=data_loader.num_workers)
    F.to(device)
    F.eval()

    pred_full = []
    with torch.no_grad():
        for data in x_plot_loader:
            data = data[0]
            data = data.to(device)
            output = F(data)
            _, pred = output.max(1)
            pred_full += pred.cpu().tolist()
    zz = np.array(pred_full).reshape(xx.shape)
    
    if newfig:
        plt.figure(figsize=(5, 5))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.axis([np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)])
        for i in range(num_class):
            mask = (y == i)
            plt.plot(X[mask][:, 0], X[mask][:, 1], '.', color=dp_colors[i])
    plt.contour(xx, yy, zz, num_class-1, colors=db_color, linewidths=1.5)
    
def test_dynse(test_loader, init_num_neighbors, validation_set, classifier_pool, device):
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        correct = 0
        total_loss = 0
        
        # cls_valid_matrix: #classifiers x #points_in_validation_set
        for i, cls in enumerate(classifier_pool):
            cls.to(device)
            cls.eval()
            sub_matrix = cls(torch.FloatTensor(validation_set.data).to(device)).max(1)[1] == torch.LongTensor(validation_set.target).to(device)
            cls_valid_matrix = torch.cat((cls_valid_matrix, sub_matrix.reshape(1,-1))) if i else sub_matrix.reshape(1,-1)  
            
        knn_model = KNeighborsClassifier(n_neighbors=init_num_neighbors) 
        knn_model.fit(validation_set.data, validation_set.target)
        
        for data, target in test_loader:  # batch_size is 1
            # find neighbors and find classifiers
            num_neighbors = init_num_neighbors
            keep_classifiers = []
            while not keep_classifiers and num_neighbors:
              _, neighbor_indexes = knn_model.kneighbors(data, n_neighbors=num_neighbors)
              cls_neighbor_matrix = cls_valid_matrix[:, neighbor_indexes.reshape(-1)]  
              keep = list(cls_neighbor_matrix.sum(dim=1) == num_neighbors)  # classify all neighbors correctly
              keep_classifiers = [i for i, kp in enumerate(keep) if kp]              
              num_neighbors -= 1
              
            if not keep_classifiers:  # still can not find any classifier even when #neighbors is 1
              keep_classifiers = [i for i in range(len(classifier_pool))]
              # keep_classifiers = [-1]
            
            # predict
            data, target = data.to(device), target.to(device)
            for i, kp in enumerate(keep_classifiers):
                cls = classifier_pool[kp]
                cls.to(device)
                cls.eval()
                cls_outputs = torch.cat((cls_outputs, cls(data))) if i else cls(data)
            
            # majority vote
            output = cls_outputs.sum(dim=0, keepdim=True)  # to refine
            output /= len(cls_outputs)  
            pred_labels = cls_outputs.max(dim=1)[1]
            pred = torch.mode(pred_labels,dim=0)[0]   
                                
            loss = criterion(output, target)
            total_loss += loss.item() * len(data)
            correct += pred.eq(target).sum().item()
    acc = correct / len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, acc

def split_train_valid(dataset, train_ratio=0.8):
    mask = np.zeros(len(dataset), dtype=bool)
    train_idx = np.random.choice(len(dataset), size=int(train_ratio*len(dataset)), replace=False)
    mask[train_idx] = True
    if dataset.__class__.__name__ == 'SoftmaxDataset':
        if dataset.loc is None:
            tset = SoftmaxDataset(np.array(dataset.softmax_data)[:,mask], mode=dataset.mode)
            vset = SoftmaxDataset(np.array(dataset.softmax_data)[:,~mask], mode=dataset.mode)
        else:
            tset = SoftmaxDataset(np.array(dataset.softmax_data)[:,mask], loc=np.array(dataset.loc)[mask], mode=dataset.mode)
            vset = SoftmaxDataset(np.array(dataset.softmax_data)[:,~mask], loc=np.array(dataset.loc)[~mask], mode=dataset.mode)            
    elif dataset.__class__.__name__ == 'BufferDataset':
        tset = BufferDataset(np.array(dataset.data)[mask], np.array(dataset.target)[mask], target_type=dataset.target_type)
        vset = BufferDataset(np.array(dataset.data)[~mask], np.array(dataset.target)[~mask], target_type=dataset.target_type)
    else:
        tset = BufferDataset(np.array(dataset.data)[mask], np.array(dataset.target)[mask], target_type='hard')
        vset = BufferDataset(np.array(dataset.data)[~mask], np.array(dataset.target)[~mask], target_type='hard')

    return tset, vset


def get_correct_status(classifier, t_data_loader, device):
    classifier.to(device), classifier.eval()
    correct_status = []
    with torch.no_grad():
        for data, target in t_data_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            _, pred = output.max(1)
            for i in range(target.shape[0]):
                correct_status.append((pred[i] == target[i]).item())
    return correct_status

def Q_statistic(correct_status_i, correct_status_j, epsilon):
    N11 = 0 
    N10 = 0 
    N01 = 0
    N00 = 0
    for k in range(len(correct_status_i)):
        if correct_status_i[k] == 1 and correct_status_j[k] == 1:
            N11 += 1
        elif correct_status_i[k] == 1 and correct_status_j[k] == 0:
            N10 += 1
        elif correct_status_i[k] == 0 and correct_status_j[k] == 1:
            N01 += 1
        else:
            N00 += 1
    return (N11 * N00 - N01 * N10) / (N11 * N00 + N01 * N10 + epsilon)

def diversity(Q_map, index):
    sum_q = 0
    total_num = 0
    for i in range(len(Q_map)):
        if i == index:
            continue
        for j in range(i + 1, len(Q_map[i])):
            if j == index:
                continue
            sum_q += Q_map[i][j]
            total_num += 1
    return 1 - (sum_q / total_num)

def MSE_i(classifier, t_data_loader, device):
    square_sum = 0
    data_num = 0
    classifier.to(device), classifier.eval()
    with torch.no_grad():
        for data, target in t_data_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            output = nn.functional.softmax(output, dim=1)
            for i in range(target.shape[0]):
                square_sum += ((1 - output[i][target[i]].item()) ** 2)
                data_num += 1
    return square_sum / data_num

def MSE_gamma(t_data_loader, classes, device):
    sum_of_classes = torch.zeros(classes)
    MSE_g = 0
    for data, target in t_data_loader:
        sum_of_classes = sum_of_classes + torch.bincount(target, minlength=classes) 
    data_num = sum_of_classes.sum()
    p_of_classes = sum_of_classes / data_num
    for c in range(classes):
        MSE_g += ((p_of_classes[c]) * ((1 - p_of_classes[c]) ** 2))
    return MSE_g.item()

# for DTEL
def test_ensemble(test_loader, classifier_list, weight_list, num_classes, device):
    for classifier in classifier_list:
        classifier.to(device)
        classifier.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        correct = 0
        total_loss = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = 0
            weight_vote = 0
            weight_sum = 0
            for classifier, weight in zip(classifier_list, weight_list):
                out = classifier(data)
                # for calculating loss
                output += (out * weight)
                # for prediction
                _, pred = out.max(1)
                vote = nn.functional.one_hot(pred, num_classes=num_classes)
                weight_vote += vote * weight
                weight_sum += weight
            output /= weight_sum
            loss = criterion(output, target)
            total_loss += loss.item() * len(data)
            weight_vote /= weight_sum
            _, pred = weight_vote.max(1)
            correct += pred.eq(target).sum().item()
    acc = correct / len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, acc

### Test DP DTEL (weighted hard/soft voting)
def test_dp_dtel_test_ensemble(test_loader, classifier_list, w, pred_classifier_list, pred_w, num_classes, device, voting='soft'): 
    correct = 0
    keep_classifiers = classifier_list + pred_classifier_list
    keep_w = w + pred_w
    keep_w = torch.FloatTensor(keep_w).view(-1, 1).to(device)
    with torch.no_grad():
        for data, target in test_loader:  # batch_size is 1
            data, target = data.to(device), target.to(device)
            for i, cls in enumerate(keep_classifiers):
                cls.to(device)
                cls.eval()
                cls_outputs = torch.cat((cls_outputs, cls(data))) if i else cls(data) 
            if voting == 'soft':
                cls_softmax = nn.Softmax(dim=1)(cls_outputs)
                weight_vote = cls_softmax * keep_w
            elif voting == 'hard':
                _, vote = cls_outputs.max(1)
                vote = nn.functional.one_hot(vote, num_classes=num_classes)
                weight_vote = vote * keep_w
            output = weight_vote.sum(dim=0, keepdim=True)
            pred = output.argmax()
            correct += pred.eq(target).sum().item()
    acc = correct / len(test_loader.dataset)
    return acc

def test_dp_dtel_get_feedback_acc(test_loader, classifier_list, pred_classifier_list, device):
    keep_classifiers = classifier_list + pred_classifier_list
    keep_w = []
    for i in range(len(keep_classifiers)):
        classifier = keep_classifiers[i]
        loss, acc = test(test_loader, classifier, device)
        keep_w.append(acc)
    return keep_w[0:len(classifier_list)], keep_w[len(classifier_list):]

### Test ensemble 3 sets (weighted hard/soft voting)
def test_ensemble_3_sets(test_loader, classifier_list, w, finetuned_classifier_list, finetuned_w, pred_classifier_list, pred_w, num_classes, device, voting='soft', return_prediction=False): 
    correct = 0
    keep_classifiers = classifier_list + finetuned_classifier_list + pred_classifier_list
    keep_w = w + finetuned_w + pred_w
    keep_w = torch.FloatTensor(keep_w).view(-1, 1).to(device)
    if return_prediction:
        preds = []
    with torch.no_grad():
        for data, target in test_loader:  # batch_size is 1
            data, target = data.to(device), target.to(device)
            for i, cls in enumerate(keep_classifiers):
                cls.to(device)
                cls.eval()
                cls_outputs = torch.cat((cls_outputs, cls(data))) if i else cls(data) 
            if voting == 'soft':
                cls_softmax = nn.Softmax(dim=1)(cls_outputs)
                weight_vote = cls_softmax * keep_w
            elif voting == 'hard':
                _, vote = cls_outputs.max(1)
                vote = nn.functional.one_hot(vote, num_classes=num_classes)
                weight_vote = vote * keep_w
            output = weight_vote.sum(dim=0, keepdim=True)
            pred = output.argmax()
            correct += pred.eq(target).sum().item()
            if return_prediction:
                preds.append(int(pred.item()))
    acc = correct / len(test_loader.dataset)
    if return_prediction:
        return acc, preds
    return acc

def get_feedback_acc_3_sets(test_loader, classifier_list, finetuned_classifier_list, pred_classifier_list, device):
    keep_classifiers = classifier_list + finetuned_classifier_list + pred_classifier_list
    keep_w = []
    for i in range(len(keep_classifiers)):
        classifier = keep_classifiers[i]
        loss, acc = test(test_loader, classifier, device)
        keep_w.append(acc)
    c, f = len(classifier_list), len(finetuned_classifier_list)
    return keep_w[:c], keep_w[c: c+f], keep_w[c+f:]

def train_ddgda(min_chunk_size, data_history, Q, optimizer, device):
    total_loss = 0.0
    criterion = nn.MSELoss()
    for train_indx in range(len(data_history)-1):
        batch_train, batch_test = data_history[train_indx], data_history[train_indx+1]
        optimizer.zero_grad()
        # bias column for X
        X_train = torch.column_stack((torch.FloatTensor(batch_train.data[:min_chunk_size]), torch.ones(min_chunk_size, 1))).to(device)
        y_train = torch.FloatTensor(batch_train.target[:min_chunk_size]).to(device)
        X_test = torch.column_stack((torch.FloatTensor(batch_test.data[:min_chunk_size]), torch.ones(min_chunk_size, 1) )).to(device)
        y_test = torch.FloatTensor(batch_test.target[:min_chunk_size]).to(device) 
        
        output = X_test@(torch.inverse(X_train.T*torch.sigmoid(Q)@X_train)@X_train.T*torch.sigmoid(Q)@y_train)
        loss = criterion(output, y_test)
        loss.backward()
        optimizer.step()
        total_loss += loss
    print('total_loss', total_loss.item())

def test_ddgda(min_chunk_size, batch_train, batch_test, Q, device):
    criterion = nn.MSELoss()
    num_test_examples = len(batch_test.data)
    with torch.no_grad():
        correct = 0
        total_loss = 0
        X_train = torch.column_stack((torch.FloatTensor(batch_train.data[:min_chunk_size]), torch.ones(min_chunk_size, 1))).to(device)
        y_train = torch.FloatTensor(batch_train.target[:min_chunk_size]).to(device)
        
        # testing
        for start_index in range(0, num_test_examples, min_chunk_size):
            if (start_index+min_chunk_size) <= num_test_examples:
                y_test = torch.FloatTensor(batch_test.target[start_index:start_index+min_chunk_size]).to(device)
                X_test = torch.FloatTensor(batch_test.data[start_index:start_index+min_chunk_size])
                X_test = torch.column_stack((X_test, torch.ones(min_chunk_size, 1))).to(device)
                output = X_test@(torch.inverse(X_train.T*torch.sigmoid(Q)@X_train)@X_train.T*torch.sigmoid(Q)@y_train)  
            else:  # the last data
                X_test = torch.FloatTensor(batch_test.data[-1*min_chunk_size:])
                X_test = torch.column_stack((X_test, torch.ones(min_chunk_size, 1))).to(device)
                output = X_test@(torch.inverse(X_train.T*torch.sigmoid(Q)@X_train)@X_train.T*torch.sigmoid(Q)@y_train)  
                
                unmask_from = -1*(num_test_examples-start_index)
                output = output[unmask_from:]
                y_test = torch.FloatTensor(batch_test.target[unmask_from:]).to(device)
  
            loss = criterion(output, y_test)        
            total_loss += loss.item() * output.shape[0]
            pred = (output>=0.5).float()  ### binarize
            correct += pred.eq(y_test).sum().item()
            
            outputs = torch.cat((outputs, output)) if start_index else output
            hard_pred = torch.cat((hard_pred, pred)) if start_index else pred
        
    acc = correct / num_test_examples
    total_loss /= num_test_examples
    
    soft_pred = torch.clamp(outputs, min=0.0, max=1.0).reshape(-1,1) # [0.3, 0.6, 0.7]
    soft_pred = torch.column_stack( ( torch.ones_like(soft_pred) - soft_pred, soft_pred) )

    return total_loss, acc, soft_pred, hard_pred
