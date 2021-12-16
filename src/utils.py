import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

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
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = nn.functional.log_softmax(DP(data), dim=1)
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
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = DP(data)
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
        for data in test_loader:
            data = data.to(device)
            prob = nn.functional.softmax(DP(data), dim=1).tolist()
            softmax_log += prob

    return softmax_log

def draw_decision_boundary(data_loader, F, device, x_range=None, y_range=None, newfig=False, db_color='b'):
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
    x_plot_loader = Data.DataLoader(x_plot_dataset, batch_size=data_loader.batch_size,
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
        mask = (y == 0)
        plt.plot(X[mask][:, 0], X[mask][:, 1], '.', color='#FF9999')
        plt.plot(X[~mask][:, 0], X[~mask][:, 1], '.', color='#99CCFF')
    plt.contour(xx, yy, zz, 0, colors=db_color, linewidths=1.5)
