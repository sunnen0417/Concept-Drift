import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from datasets import BufferDataset, SoftmaxDataset, SoftmaxOnlineDataset
from sklearn.neighbors import KNeighborsClassifier
import torch.optim as optim

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

class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.params = self.shadow.keys()

    def __call__(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if name in self.params and param.requires_grad:
                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
                    param.data = self.shadow[name]

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, logit):
        b = nn.functional.softmax(logit, dim=1) * nn.functional.log_softmax(logit, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

class VAT(nn.Module):
    def __init__(self, XI, perturb_radius, model, device):
        super(VAT, self).__init__()
        self.device = device
        self.n_power = 1  
        self.XI = XI
        self.model = model
        self.epsilon = perturb_radius

    def forward(self, X, logit):
        vat_loss = self.virtual_adversarial_loss(X, logit)
        return vat_loss

    def generate_virtual_adversarial_perturbation(self, x, logit):
        d = torch.randn_like(x).to(self.device)

        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            logit_m = self.model(x + d)
            dist = self.kl_divergence_with_logit(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = nn.functional.softmax(q_logit, dim=1)
        qlogq = torch.mean(torch.sum(q * nn.functional.log_softmax(q_logit, dim=1), dim=1))
        qlogp = torch.mean(torch.sum(q * nn.functional.log_softmax(p_logit, dim=1), dim=1))
        return qlogq - qlogp

    def get_normalized_vector(self, d):
        return nn.functional.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)
        logit_p = logit.detach()
        logit_m = self.model(x + r_vadv)
        loss = self.kl_divergence_with_logit(logit_p, logit_m)
        return loss
    
# Training using soft label + DIRT-T cluster assumption
def train_soft_dirt(train_loader, F, optimizer, F_old, radius, lamb, beta, device):
    F.to(device)
    F.train()
    F_old.to(device)
    F_old.eval()
    criterion1 = nn.KLDivLoss(reduction='batchmean')
    criterion2 = ConditionalEntropyLoss()
    criterion3 = VAT(1e-6, radius, F, device)
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logit = F(data)
        logit_old = F_old(data)
        logit_old = logit_old.detach()
        output = nn.functional.log_softmax(logit, dim=1)
        loss1 = criterion1(output, target)
        loss2 = criterion2(logit)
        loss3 = criterion3(data, logit)
        loss4 = criterion1(output, nn.functional.softmax(logit_old, dim=1))
        loss = loss1 + lamb * (loss2 + loss3) + beta * loss4
        loss.backward()
        optimizer.step()

# Testing using soft label + DIRT-T cluster assumption
def test_soft_dirt(test_loader, F, F_old, radius, lamb, beta, device, return_softmax=False):
    F.to(device)
    F.eval()
    F_old.to(device)
    F_old.eval()
    criterion1 = nn.KLDivLoss(reduction='batchmean')
    criterion2 = ConditionalEntropyLoss()
    criterion3 = VAT(1e-6, radius, F, device)
    if return_softmax:
        softmax_log = []
    total_loss = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logit = F(data)
        logit_old = F_old(data)
        logit_old = logit_old.detach()
        output = nn.functional.log_softmax(logit, dim=1)
        loss1 = criterion1(output, target)
        loss2 = criterion2(logit)
        loss3 = criterion3(data, logit)
        loss4 = criterion1(output, nn.functional.softmax(logit_old, dim=1))
        loss = loss1 + lamb * (loss2 + loss3) + beta * loss4
        total_loss += loss.item() * len(data)
        if return_softmax:
            prob = nn.functional.softmax(logit, dim=1).tolist()
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
    

# Losses for VAE
    
# reconstruction loss
class ReconstructionLoss(nn.Module):
    def __init__(self, cate_feat=[], reduction=True):
        super(ReconstructionLoss, self).__init__()
        self.cate_feat = cate_feat
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.binary_cate_feat = []
        self.cate_feat_flatten = []
        for l in self.cate_feat:
            if len(l) == 1:
                self.binary_cate_feat += l
            self.cate_feat_flatten += l

    def forward(self, output, target): 
        # output: reconstructed samples; target: real samples
        total_loss = 0
        # binary cate. features
        if len(self.binary_cate_feat) > 0:
            total_loss += torch.sum(self.bce_loss(output[:, self.binary_cate_feat], target[:, self.binary_cate_feat]), dim=1)
        # cate. features
        for l in self.cate_feat:
            if len(l) > 1:
                labels = torch.argmax(target[:, l], dim=1)
                total_loss += self.cross_entropy(output[:, l], labels)
        # numerical features
        numeric_idx = [i for i in range(output.size(1)) if i not in self.cate_feat_flatten]
        total_loss += torch.sum(self.mse_loss(output[:, numeric_idx], target[:, numeric_idx]), dim=1)

        if self.reduction:
            return torch.mean(total_loss)
        else:
            return total_loss
    
# extension loss
class ExtentionLoss(nn.Module):
    def __init__(self, reduction=True):
        super(ExtentionLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.reduction = reduction
        
    def forward(self, output): 
        # output: logits
        output = output.view(-1)
        p = self.sigmoid(output)
        if self.reduction:
            return torch.mean(1-p)
        else:
            return 1 - p

# kl divergence loss
class KLDLoss(nn.Module):
    def __init__(self, reduction=True):
        super(KLDLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, mu, logvar):
        # mu: mean; logvar: log of variance
        if self.reduction:
            return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        else:
            return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1)
    
# wrap loss
class WrapLoss(nn.Module):
    def __init__(self, cate_feat=[], reduction=False):
        super(WrapLoss, self).__init__()
        self.cate_feat = cate_feat
        self.reduction = reduction
        self.r_loss = ReconstructionLoss(self.cate_feat, self.reduction)
        self.e_loss = ExtentionLoss(self.reduction)
        
    def forward(self, output, target, extension):
        # output: reconstructed samples; target: real samples; extension: logits
        d = output.size(1)
    
        return self.r_loss(output, target) / d + self.e_loss(extension)

# loss for positive samples
class PositiveLoss(nn.Module):
    def __init__(self, cate_feat=[], reduction=True):
        super(PositiveLoss, self).__init__()
        self.cate_feat = cate_feat
        self.reduction = reduction
        self.r_loss = ReconstructionLoss(self.cate_feat, self.reduction)
        self.e_loss = ExtentionLoss(self.reduction)
        self.k_loss = KLDLoss(self.reduction)
        
    def forward(self, output, target, extension, mu, logvar):
        # output: reconstructed samples; target: real samples; extension: logits; mu: mean; logvar: log of variance
        return self.r_loss(output, target) + self.e_loss(extension) + self.k_loss(mu, logvar)
        
# loss for negative samples
class NegativeLoss(nn.Module):
    def __init__(self, reduction=True):
        super(NegativeLoss, self).__init__()
        self.reduction = reduction
        self.e_loss = ExtentionLoss(self.reduction)
        
    def forward(self, extension):
        # extension: logits
        return self.e_loss(extension)
    
# Total cost (reduction must be True)
class JLoss(nn.Module):
    def __init__(self, cate_feat=[]):
        super(JLoss, self).__init__()
        self.cate_feat = cate_feat
        self.p_loss = PositiveLoss(cate_feat=self.cate_feat, reduction=True)
        self.n_loss = NegativeLoss(reduction=True)
        
    def forward(self, output, target, extension, mu, logvar, label):
        # output: reconstructed samples; target: real samples; extension: logits; mu: mean; logvar: log of variance
        # label: 0 for negative, 1 for positive
        i = label == 1
        if len(extension[i]) == 0:
            p = 0
        else:
            p = self.p_loss(output[i], target[i], extension[i], mu[i], logvar[i])
        
        j = label == 0
        if len(extension[j]) == 0:
            n = 0
        else:
            n = self.n_loss(extension[j])
        
        return p - n
         
# Train VAE
def train_vae(train_loader, vae, optimizer, cate_feat, device):
    vae.to(device)
    vae.train()
    criterion = JLoss(cate_feat=cate_feat)
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        feature, extension, mu, logvar = vae(data, sample=True)
        loss = criterion(feature, data, extension, mu, logvar, target)
        loss.backward()
        optimizer.step()
        
# Test VAE
def test_vae(test_loader, vae, theta, cate_feat, device):
    vae.to(device)
    vae.eval()
    criterion1 = JLoss(cate_feat=cate_feat)
    criterion2 = WrapLoss(cate_feat=cate_feat, reduction=False)
    isPass = True
    with torch.no_grad():
        total_loss1 = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            feature, extension, mu, logvar = vae(data, sample=False)
            loss1 = criterion1(feature, data, extension, mu, logvar, target)
            total_loss1 += loss1.item()
            loss2 = criterion2(feature, data, extension)
            for i in range(len(target)):
                if target[i] == 1:
                    if loss2[i] >= theta:
                        isPass = False
                    else:
                        correct += 1
                else:
                    if loss2[i] < theta:
                        isPass = False
                    else:
                        correct += 1
    return total_loss1 * test_loader.batch_size / len(test_loader.dataset), correct / len(test_loader.dataset), isPass

# Train VAE with several epochs
def train_epochs_vae(train_loader, vae, optimizer, epochs, theta, cate_feat, device):
    for i in range(epochs):
        print('Epoch:', i+1)
        train_vae(train_loader, vae, optimizer, cate_feat, device)
        loss, acc, isPass = test_vae(train_loader, vae, theta, cate_feat, device)
        print(f'loss:{loss}, acc:{acc}')
        if isPass:
            print('Early stopping!')
            break

# Sample data from VAE
def sample(vae, theta, sample_n, cate_feat, device, batch_size=64, num_workers=0):
    vae.to(device)
    vae.eval()
    criterion = WrapLoss(cate_feat=cate_feat, reduction=False)
    sigmoid = nn.Sigmoid()
    n = sample_n * 2
    d = vae.latent_size
    a = np.random.randn(n, d)
    b = np.zeros(n)
    dataset = BufferDataset(a, b, target_type='hard')
    data_loader = Data.DataLoader(dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    data_samples = []
    with torch.no_grad():
        for r, _ in data_loader:
            r = r.to(device)
            feature, extension = vae.decode(r)
            for l in cate_feat:
                if len(l) == 1:
                    sigmoid_feature = sigmoid(feature[:, l])
                    idx = sigmoid_feature >= 0.5
                    sigmoid_feature[idx] = 1.0
                    sigmoid_feature[~idx] = 0.0
                    feature[:, l] = sigmoid_feature
                else:
                    idx = torch.argmax(feature[:, l], dim=1)
                    one_hot_feature = torch.zeros_like(feature[:, l])
                    one_hot_feature[np.arange(one_hot_feature.size(0)), idx]
                    feature[:, l] = one_hot_feature

            feature2, extension2, mu, logvar = vae(feature, sample=False)
            loss = criterion(feature2, feature, extension2)
            idx = loss < theta
            l = feature[idx].tolist()
            data_samples += l
    data_samples = np.array(data_samples)
    return data_samples[0:sample_n]


# Update all VAEs with the dataset
def update_all_vaes(train_loader, vae_list, optimizer_list, epochs, theta, sample_n, eps, device):
    M = len(vae_list)
    trainset = train_loader.dataset
    batch_size = train_loader.batch_size
    num_workers = train_loader.num_workers
    cate_feat = trainset.cate_feat
    for i in range(M):
        # extract class i data
        idx = trainset.target == i
        data = trainset.data[idx]
        target = np.zeros(len(data))
        S = BufferDataset(data, target, target_type='hard')
        if len(S) == 0:
            continue
        print(f'Processing data with class {i}')
        # uncover j-th vae where j != i
        for j in range(M):
            if j != i:
                data_loader = Data.DataLoader(S, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers)
                loss, acc, isPass = test_vae(data_loader, vae_list[j], theta, cate_feat, device)
                if not isPass:
                    print(f'Uncovering VAE {j} on data with class {i}')
                    # positive samples
                    S_samples = S.data.numpy()
                    wrap_samples = sample(vae_list[j], theta, sample_n, cate_feat, device, batch_size=batch_size, num_workers=num_workers)
                    toCollect = []
                    for k in range(len(wrap_samples)):
                        dist = np.linalg.norm(S_samples-wrap_samples[k], axis=1)
                        a = dist < eps
                        if True in a:
                            continue
                        else:
                            toCollect.append(k)
                    toCollect = np.array(toCollect)
                    if len(toCollect) == 0:
                        positive_samples = np.array([])
                    else:
                        positive_samples = wrap_samples[toCollect]
                    # negative samples
                    negative_samples = []
                    for k in range(M):
                        if k != j:
                            wrap_samples = sample(vae_list[k], theta, sample_n, cate_feat, device, batch_size=batch_size, num_workers=num_workers)
                            negative_samples += wrap_samples.tolist()
                    negative_samples += S_samples.tolist()
                    negative_samples = np.array(negative_samples)
                    
                    samples = positive_samples.tolist() + negative_samples.tolist()
                    samples = np.array(samples)
                    labels = np.zeros(len(samples))
                    labels[:len(positive_samples)] = 1
                    uncover_dataset = BufferDataset(samples, labels, target_type='hard')
                    data_loader = Data.DataLoader(uncover_dataset, batch_size=batch_size, 
                                            shuffle=True, num_workers=num_workers)
                    train_epochs_vae(data_loader, vae_list[j], optimizer_list[j], epochs, theta, cate_feat, device)
        # cover i-th vae
        # positive samples
        print(f'Covering VAE {i} on data with class {i}')
        S_samples = S.data.numpy()
        wrap_samples = sample(vae_list[i], theta, sample_n, cate_feat, device, batch_size=batch_size, num_workers=num_workers)
        positive_samples = S_samples.tolist() + wrap_samples.tolist()
        positive_samples = np.array(positive_samples)
        # negative samples
        negative_samples = []
        for j in range(M):
            if j != i:
                wrap_samples = sample(vae_list[j], theta, sample_n, cate_feat, device, batch_size=batch_size, num_workers=num_workers)
                negative_samples += wrap_samples.tolist()
        negative_samples = np.array(negative_samples)
        
        samples = positive_samples.tolist() + negative_samples.tolist()
        samples = np.array(samples)
        labels = np.zeros(len(samples))
        labels[:len(positive_samples)] = 1
        cover_dataset = BufferDataset(samples, labels, target_type='hard')
        data_loader = Data.DataLoader(cover_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)        
        train_epochs_vae(data_loader, vae_list[i], optimizer_list[i], epochs, theta, cate_feat, device)


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
    elif dataset.__class__.__name__ == 'SoftmaxOnlineDataset':
        if dataset.loc is None:
            tset = SoftmaxOnlineDataset([dataset.softmax_data[i] for i, flag in enumerate(mask) if flag], mode=dataset.mode)
            vset = SoftmaxOnlineDataset([dataset.softmax_data[i] for i, flag in enumerate(mask) if not flag], mode=dataset.mode)
        else:
            tset = SoftmaxOnlineDataset([dataset.softmax_data[i] for i, flag in enumerate(mask) if flag], loc=np.array(dataset.loc)[mask], mode=dataset.mode)
            vset = SoftmaxOnlineDataset([dataset.softmax_data[i] for i, flag in enumerate(mask) if not flag], loc=np.array(dataset.loc)[~mask], mode=dataset.mode)            
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

# bootstrapping for softmax_log
def bootstrap(softmax_log):
    s = np.array(softmax_log)
    n = s.shape[1]
    idx = np.random.choice(n, size=n, replace=True)
    return s[:,idx].tolist()

### DTEL_DP (soft vote)
def dtel_test_ensemble(test_loader, classifier_pool, pred_classifier_pool, device): 
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:  # batch_size is 1
            keep_classifiers = classifier_pool + pred_classifier_pool 
            data, target = data.to(device), target.to(device)
            for i, cls in enumerate(keep_classifiers):
                cls.to(device)
                cls.eval()
                cls_outputs = torch.cat((cls_outputs, cls(data))) if i else cls(data) 
            
            cls_softmax = nn.Softmax(dim=1)(cls_outputs)
            output = cls_softmax.mean(dim=0, keepdim=True)
            pred = output.argmax()
                      
            loss = criterion(output, target)
            total_loss += loss.item() * len(data)
            correct += pred.eq(target).sum().item()
    acc = correct / len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, acc
    

### DTEL_DP (cluster assumption weighted vote)  
def refine_ca(ca_epochs, ca_lr, ema_decay, XI, perturb_radius, data_loader, F, device):
    F.to(device)
    F.train()
    optimizer = optim.Adam(F.parameters(), lr=ca_lr)
    cent = ConditionalEntropyLoss().to(device)
    vat_loss = VAT(XI, perturb_radius, F, device).to(device)
    ema = EMA(ema_decay)
    ema.register(F)
    
    for epoch in range(ca_epochs):
        for data, _ in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = F(data)
            loss_cent = cent(output)
            loss_vat = vat_loss(data, output)
            loss = loss_cent + loss_vat 
            
            loss.backward()
            optimizer.step()
            
            ema(F)
            
def get_ensemble_ca_weight(cls_neighbor_softmax, epsilon=1e-8):
    weight = [] 
    ca_softmax = cls_neighbor_softmax[-1]
    for i, c_softmax in enumerate(cls_neighbor_softmax[:-1]):
        # for each neighbor
        for j, (c_soft, ca_soft) in enumerate(zip(c_softmax, ca_softmax)):
            if not j:
                dist = nn.KLDivLoss()(c_soft.log(), ca_soft).item() 
            else:
                dist += nn.KLDivLoss()(c_soft.log(), ca_soft).item() 
        weight.append(1.0 / (dist + epsilon))
    
    weight = nn.functional.softmax(torch.FloatTensor(weight), dim=0)
    return weight
    
def dtel_test_ensemble_ca(test_loader, validation_set, F_ca, classifier_pool, pred_classifier_pool, num_neighbors, device): 
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        correct = 0
        total_loss = 0
               
        for i, cls in enumerate(classifier_pool+pred_classifier_pool+[F_ca]):
            cls.to(device)
            cls.eval()
            with torch.no_grad():
                valid_logit = cls(torch.FloatTensor(validation_set.data).to(device)) 
                valid_softmax = nn.Softmax(dim=1)(valid_logit)
                cls_valid_softmax = torch.cat((cls_valid_softmax, valid_softmax.unsqueeze(0))) if i else valid_softmax.unsqueeze(0)  
                
        knn_model = KNeighborsClassifier(n_neighbors=num_neighbors) 
        knn_model.fit(validation_set.data, validation_set.target)
        
        for data, target in test_loader:  # batch_size is 1
            # find neighbors
            _, neighbor_indexes = knn_model.kneighbors(data, n_neighbors=num_neighbors)
            # get {all c_softmax for c in E & E' and ca_softmax} of each neighbor
            cls_neighbor_softmax = cls_valid_softmax[:, neighbor_indexes.reshape(-1),:]  
            cls_weight = get_ensemble_ca_weight(cls_neighbor_softmax).reshape(-1,1).to(device)
            
            # predict
            keep_classifiers = classifier_pool + pred_classifier_pool
            data, target = data.to(device), target.to(device)
            for i, cls in enumerate(keep_classifiers):
                cls.to(device)
                cls.eval()
                cls_outputs = torch.cat((cls_outputs, cls(data))) if i else cls(data)
                
            # weighted vote
            cls_softmax = nn.Softmax(dim=1)(cls_outputs)
            weighted_softmax = cls_weight * cls_softmax
            weighted_sum = weighted_softmax.sum(dim=0, keepdim=True)
            
            output = weighted_sum/cls_weight.sum()
            pred_labels = weighted_sum.argmax()
                                
            loss = criterion(output, target)
            total_loss += loss.item() * len(data)
            correct += pred_labels.eq(target).sum().item()
            
    acc = correct / len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, acc