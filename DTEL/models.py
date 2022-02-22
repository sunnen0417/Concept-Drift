import math
import random
import numpy as np
import collections
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from operator import itemgetter
from functools import reduce
from sklearn.neighbors import KNeighborsClassifier

# logistic regression classifier
class LogisticRegression(nn.Module):
    def __init__(self, in_size=2, out_size=2):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, x):
        x = self.fc(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_size=2, out_size=2):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, 16 * in_size),
            nn.ReLU(),
            nn.Linear(16 * in_size, out_size)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DynamicPredictor(nn.Module):
  def __init__(self, in_size=2, d_model=4, dropout=0.1):
    super(DynamicPredictor, self).__init__()
    # Project the dimension of features from that of input into d_model.
    self.prenet = nn.Linear(in_size, d_model)
    self.pos_encoder = PositionalEncoding(d_model, dropout)
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, dim_feedforward=d_model*4, nhead=1, dropout=dropout
    )
    # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    # Project the the dimension of features from d_model into speaker nums.
    self.pred_layer = nn.Sequential(
      nn.Linear(d_model, in_size),
    )

  def forward(self, x):
    """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
    # out: (batch size, length, d_model)
    out = self.prenet(x)
    # out: (length, batch size, d_model)
    out = out.permute(1, 0, 2)
    # Positional encoding
    out = self.pos_encoder(out)
    # The encoder layer expect features in the shape of (length, batch size, d_model).
    out = self.encoder_layer(out)
    # out: (batch size, length, d_model)
    out = out.transpose(0, 1)
    # mean pooling
    stats = out.mean(dim=1)

    # out: (batch, n_spks)
    out = self.pred_layer(stats)
    return out

class VAE(nn.Module):
    def __init__(self, feat_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.latent_size= latent_size
        # encoder
        self.e1 = nn.Linear(feat_size, hidden_size//2)
        self.e2 = nn.Linear(hidden_size//2, hidden_size)
        self.e3_mu = nn.Linear(hidden_size, latent_size)
        self.e3_logvar = nn.Linear(hidden_size, latent_size)
        # decoder
        self.d1 = nn.Linear(latent_size, hidden_size//2)
        self.d2 = nn.Linear(hidden_size//2, hidden_size)
        self.d3 = nn.Linear(hidden_size, feat_size+1) # dimension extension
        self.relu = nn.ReLU()
    
    def encode(self, x):
        x = self.relu(self.e1(x))
        x = self.relu(self.e2(x))
        mu = self.e3_mu(x)
        logvar = self.e3_logvar(x)
        return mu, logvar.clamp(-5, 5)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def decode(self, x):
        x = self.relu(self.d1(x))
        x = self.relu(self.d2(x))
        x = self.d3(x)
        feature, extension = x[:, :self.feat_size], x[:, self.feat_size:]
        return feature, extension
    
    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        if sample:
            z = self.reparameterize(mu, logvar)
            feature, extension = self.decode(z)
        else:
            feature, extension = self.decode(mu)
        return feature, extension, mu, logvar
    
    def extension_logit(self, x):
        mu, logvar = self.encode(x)
        extension = self.decode(mu)[1]
        return extension

# Subspace buffer
class ReplayBuffer(ABC):

    @abstractmethod
    def add(self, x_batch, y_batch, weights, **kwargs):
        pass

    @abstractmethod
    def sample(self, x_batch, y_batch, weights, **kwargs):
        pass

class SubspaceBuffer(ReplayBuffer):
    def __init__(self, max_centroids: int, max_instances: int, centroids_frac: float=1.0):
        self.max_centroids = max_centroids
        self.max_instances = max_instances
        self.centroids_frac = centroids_frac

        self.centroids = collections.defaultdict(list)
        self.total_num_centroids = 0
        self.buffers = collections.defaultdict(list)

    def add(self, x_batch, y_batch, weights, **kwargs):
        if torch.is_tensor(x_batch):
            x_batch, y_batch, weights = x_batch.cpu().numpy(), y_batch.cpu().numpy(), weights.cpu().numpy()

        for x, y, w in zip(x_batch, y_batch, weights):
            if len(self.centroids[y]) == self.max_centroids:
                centroid_idx, _ = min([(i, np.linalg.norm(x - p[0])) for i, p in enumerate(self.centroids[y])], key=itemgetter(1))
                mean, w_sum, var = self.centroids[y][centroid_idx]
                new_mean = mean + (w / (w_sum + w)) * (x - mean)
                new_w_sum = w_sum + w
                new_var = var + w * np.multiply(x - mean, x - new_mean)
                self.centroids[y][centroid_idx] = (new_mean, new_w_sum, new_var)

                if len(self.buffers[y][centroid_idx]) == self.max_instances:
                    self.buffers[y][centroid_idx].pop(0)

                self.buffers[y][centroid_idx].append((x, y, w))
            else:
                self.centroids[y].append((x, w, np.zeros(len(x))))
                self.total_num_centroids += 1
                self.buffers[y].append([(x, y, w)])

    def sample(self, x_batch, y_batch, weights, **kwargs):
        num_samples_per_instance = int(self.centroids_frac * self.total_num_centroids)
        num_samples = num_samples_per_instance * len(x_batch)

        input_shape = x_batch[0].shape
        sampled_x_batch = np.zeros((num_samples, *input_shape))
        sampled_y_batch = np.zeros(num_samples)
        sampled_weights = np.zeros(num_samples)

        i = 0
        centroids_buffers = list(reduce(lambda a, b: a + b, self.buffers.values(), []))
        random.shuffle(centroids_buffers)

        for _ in range(len(x_batch)):
            for j in range(num_samples_per_instance):
                (rx, ry, rw) = random.choice(centroids_buffers[j])
                sampled_x_batch[i, :] = rx[:]
                sampled_y_batch[i] = ry
                sampled_weights[i] = rw
                i += 1

        return sampled_x_batch, sampled_y_batch, sampled_weights
    
    def pour(self):
        data = []
        target = []
        for y, class_buffers in self.buffers.items():
            for class_buffer in class_buffers:
                for x, y, w in class_buffer:
                    data.append(x)
                    target.append(y)
        data = np.array(data)
        target = np.array(target)
        return data, target
    
    def region(self, x):
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        has_cluster_nearby = False
        for y, class_centroids in self.centroids.items():
            for mean, w_sum, var in class_centroids:
                dist = np.linalg.norm(x-mean)
                std = math.sqrt(var.mean()/w_sum)
                if dist / math.sqrt(x.size) <= std:
                    has_cluster_nearby = True
                    break
            if has_cluster_nearby:
                break
        return has_cluster_nearby

# Dynse + DP model
class DynseDP:
    def __init__(self, w, e, b, k, device):
        # w: maximum window size, e: maximum ensemble size, b: maximum buffer size, k: k for knn
        # device: device when testing
        self.w = w
        self.e = e
        self.b = b
        self.k = k
        self.device = device
        self.ensemble = []
        self.buffers = []
        
    def add_classifier(self, classifier):
        classifier.cpu()
        classifier.eval()
        self.ensemble.append(classifier)
        if len(self.ensemble) > self.e:
            self.ensemble.pop(0)
            
    def add_buffer(self, buffer):
        self.buffers.append(buffer)
        if len(self.buffers) > self.b:
            self.buffers.pop(0)    
            
    def trace(self, x):
        x = torch.FloatTensor(x)
        softmax_log = []
        start = max(-self.w, -len(self.buffers)) 
        end = -1
        for i in range(start, end+1):
            if self.buffers[i].region(x):
                classifier = self.ensemble[i]
                classifier.to(self.device)
                with torch.no_grad():
                    output = classifier(x.to(self.device))
                    prob = nn.functional.softmax(output, dim=-1).tolist()
                    softmax_log.append(prob)
                classifier.cpu()
            else:
                if len(softmax_log) == 0:
                    continue
                else:
                    softmax_log.append(softmax_log[-1])
        return softmax_log
    
    def predict(self, x):
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        data = []
        target = []
        for buffer in self.buffers:
            X, y = buffer.pour()
            data += list(X)
            target += list(y)
        data = np.array(data)
        target = np.array(target)
        neigh = KNeighborsClassifier(n_neighbors=self.k)
        neigh.fit(data, target)
        neigh_dist, neigh_ind = neigh.kneighbors([x], n_neighbors=self.k)
        neigh_data = torch.FloatTensor(data[neigh_ind[0]])
        neigh_target = torch.LongTensor(target[neigh_ind[0]])
        chosen = []
        for k in range(self.k, -1, -1):
            for i in range(len(self.ensemble)):
                classifier = self.ensemble[i]
                classifier.to(self.device)
                with torch.no_grad():
                    output = classifier(neigh_data.to(self.device))
                    _, y_pred = output.max(1)
                    cmp = y_pred.cpu() == neigh_target
                    correct = np.sum(cmp.numpy())
                if correct == k:
                    chosen.append(i)
                classifier.cpu()
            if len(chosen) > 0:
                break
        ensemble_pred = []
        for i in chosen:
            classifier = self.ensemble[i]
            classifier.to(self.device)
            with torch.no_grad():
                output = classifier(torch.FloatTensor(x).to(self.device))
                y_pred = torch.argmax(output).item()
                ensemble_pred.append(y_pred)
            classifier.cpu()
        u, cnt = np.unique(ensemble_pred, return_counts=True)
        return u[np.argmax(cnt)]
