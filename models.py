import math
import copy
import random
import numpy as np
import collections
import torch
import torch.nn as nn
from collections import Counter
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
  def __init__(self, in_size=2, d_model=4, dropout=0.1, location=False, location_dim=None, location_weight=0.1):
    super(DynamicPredictor, self).__init__()
    # Project the dimension of features from that of input into d_model.
    self.prenet = nn.Linear(in_size, d_model)
    self.pos_encoder = PositionalEncoding(d_model, dropout)
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, dim_feedforward=d_model*4, nhead=1, dropout=dropout
    )
    # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    # Project the the dimension of features from d_model into in_size.
    self.pred_layer = nn.Sequential(
      nn.Linear(d_model, in_size),
    )

    self.location = location
    self.location_weight = location_weight

    if location:
        assert(location_dim != None)
        self.location_layer = MLP(location_dim, in_size)

  def forward(self, x, location_info=None):
    """
    args:
      x: (batch size, length, in_size)
    return:
      out: (batch size, in_size)
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

    # out: (batch, in_size)
    out = self.pred_layer(stats)

    if self.location:
        assert(location_info != None)
        location_bias = self.location_layer(location_info)
        
        return (1 - self.location_weight) * out + self.location_weight * location_bias

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


class ReplayBuffer(ABC):

    @abstractmethod
    def add(self, x_batch, y_batch, weights, **kwargs):
        pass

    @abstractmethod
    def sample(self, x_batch, y_batch, weights, **kwargs):
        pass

# Subspace buffer
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

# Reactive subspace buffer
class ReactiveSubspaceBuffer(ReplayBuffer):

    def __init__(self, max_centroids: int, max_instances: int, window_size: int=100, switch_thresh: float=0.9, split=False,
                 split_thresh: float=0.5, split_period: int=1000):
        super().__init__()
        self.max_centroids = max_centroids
        self.max_instances = max_instances
        self.window_size = window_size
        self.switch_thresh = switch_thresh
        self.split = split
        self.split_thresh = split_thresh
        self.split_period = split_period

        self.splits_num = 0
        self.switches_num = 0

        self.centroids = collections.defaultdict(lambda: collections.defaultdict(tuple))
        self.total_num_centroids = 0
        self.buffers = collections.defaultdict(lambda: collections.defaultdict(list))

        self.centroids_window_counts = collections.defaultdict(lambda: collections.defaultdict(Counter))
        self.centroids_window_buffers = collections.defaultdict(lambda: collections.defaultdict(list))
        self.centroids_window_last_update = collections.defaultdict(lambda: collections.defaultdict(int))
        self.t = 0

        self.next_centroid_idx = 0

    def add(self, x_batch, y_batch, weights, **kwargs):
        self.t += len(x_batch)
        
        if torch.is_tensor(x_batch):
            x_batch, y_batch, weights = x_batch.cpu().numpy(), y_batch.cpu().numpy(), weights.cpu().numpy()

        for x, y, w in zip(x_batch, y_batch, weights):
            if len(self.centroids[y]) < self.max_centroids / 2:
                self.__add_centroid(x, y, w)
                continue

            closest_centroid_idx, closest_centroid_y, dist = self.__find_closest_centroid(x)

            if closest_centroid_y == y:
                self.__update_centroid(x, y, w, closest_centroid_y, closest_centroid_idx)
                self.__update_centroid_window(x, y, w, closest_centroid_y, closest_centroid_idx)
            else:
                w_sum, var = self.centroids[closest_centroid_y][closest_centroid_idx][1:]
                std = math.sqrt(var.mean() / w_sum)

                if dist / math.sqrt(x.size) <= std:
                    window_buffer = self.__update_centroid_window(x, y, w, closest_centroid_y, closest_centroid_idx)

                    if len(window_buffer) == self.window_size:
                        centroid_switch, max_class = self.__check_centroid_switch(closest_centroid_y, closest_centroid_idx)
                        if centroid_switch:
                            print('Actual drift happens!')
                            self.switches_num += 1
                            self.__switch_centroid(closest_centroid_y, closest_centroid_idx, max_class)
                else:
                    closest_y_centroid_idx, y, dist = self.__find_closest_centroid(x, y)
                    w_sum, var = self.centroids[y][closest_y_centroid_idx][1:]
                    std = math.sqrt(var.mean() / w_sum)

                    if dist / math.sqrt(x.size) <= std or len(self.centroids_window_buffers[y][closest_y_centroid_idx]) < self.window_size \
                            or len(self.centroids[y]) >= self.max_centroids:
                        self.__update_centroid(x, y, w, y, closest_y_centroid_idx)
                        self.__update_centroid_window(x, y, w, y, closest_y_centroid_idx)
                    else:
                        self.__add_centroid(x, y, w)

        if self.split:
            self.__check_centroids()

    def __add_centroid(self, x, y, w):
        self.centroids[y][self.next_centroid_idx] = (x, w, np.zeros(x.shape))
        self.buffers[y][self.next_centroid_idx] = [(x, y, w)]
        self.centroids_window_counts[y][self.next_centroid_idx] = Counter([y])
        self.centroids_window_buffers[y][self.next_centroid_idx] = [(x, y, w)]
        self.centroids_window_last_update[y][self.next_centroid_idx] = self.t
        self.total_num_centroids += 1
        self.next_centroid_idx += 1

    def __find_closest_centroid(self, x, y=-1):
        closest_centroid_idx, closest_centroid_y, min_dist = -1, -1, float('inf')
        centroids = self.centroids.items() if y < 0 else [(y, self.centroids[y])]

        for cy, class_centroids in centroids:
            if len(class_centroids) == 0:
                continue

            centroid_idx, dist = min([(centroid_idx, np.linalg.norm(x - cv[0])) for centroid_idx, cv in class_centroids.items()], key=itemgetter(1))
            if dist < min_dist:
                closest_centroid_idx = centroid_idx
                closest_centroid_y = cy
                min_dist = dist

        return closest_centroid_idx, closest_centroid_y, min_dist

    def __update_centroid_window(self, x, y, w, centroid_y, centroid_idx):
        window_buffer = self.centroids_window_buffers[centroid_y][centroid_idx]

        if len(window_buffer) == self.window_size:
            _, wy, _ = window_buffer.pop(0)
            self.centroids_window_counts[centroid_y][centroid_idx][wy] -= 1

        window_buffer.append((x, y, w))
        self.centroids_window_counts[centroid_y][centroid_idx][y] += 1
        self.centroids_window_last_update[centroid_y][centroid_idx] = self.t

        return window_buffer

    def __update_centroid(self, x, y, w, centroid_y, centroid_idx):
        mean, w_sum, var = self.centroids[centroid_y][centroid_idx]
        new_mean = mean + (w / (w_sum + w)) * (x - mean)
        new_w_sum = w_sum + w
        new_var = var + w * np.multiply(x - mean, x - new_mean)

        self.centroids[centroid_y][centroid_idx] = (
            new_mean,
            new_w_sum,
            np.array(new_var)
        )

        if len(self.buffers[centroid_y][centroid_idx]) == self.max_instances:
            self.buffers[centroid_y][centroid_idx].pop(0)

        self.buffers[centroid_y][centroid_idx].append((x, y, w))

    def __check_centroid_switch(self, centroid_y, centroid_idx):
        max_class, max_cnt = self.centroids_window_counts[centroid_y][centroid_idx].most_common(1)[0]
        current_cls_cnt = self.centroids_window_counts[centroid_y][centroid_idx].get(centroid_y)

        return current_cls_cnt / max_cnt < self.switch_thresh, max_class

    def __switch_centroid(self, centroid_y, centroid_idx, new_class):
        self.__extract_new_centroid(centroid_y, centroid_idx, new_class)
        self.__remove_centroid(centroid_y, centroid_idx)

    def __extract_new_centroid(self, centroid_y, centroid_idx, new_class, split=False):
        if centroid_y != new_class and len(self.centroids[new_class]) >= self.max_centroids:
            return
        
        window_buffer = self.centroids_window_buffers[centroid_y][centroid_idx]
        filtered_window = list(filter(lambda r: r[1] == new_class, window_buffer))

        mean, w_sum, var = filtered_window[0][2] * filtered_window[0][0], filtered_window[0][2], np.zeros(filtered_window[0][0].shape)
        for i in range(1, len(filtered_window)):
            fx, _, fw = filtered_window[i]
            pm = mean
            w_sum += fw
            mean = pm + (fw / w_sum) * (fx - pm)
            var = var + fw * np.multiply(fx - pm, fx - mean)

        self.centroids[new_class][self.next_centroid_idx] = (mean, w_sum, np.array(var))
        self.buffers[new_class][self.next_centroid_idx] = filtered_window
        self.centroids_window_counts[new_class][self.next_centroid_idx] = self.centroids_window_counts[centroid_y][centroid_idx].copy() \
            if not split else Counter({new_class: self.centroids_window_counts[centroid_y][centroid_idx].get(new_class)})
        self.centroids_window_buffers[new_class][self.next_centroid_idx] = window_buffer.copy() if not split else filtered_window.copy()
        self.centroids_window_last_update[new_class][self.next_centroid_idx] = self.t
        self.next_centroid_idx += 1
        self.total_num_centroids += 1

    def __remove_centroid(self, centroid_y, centroid_idx):
        del self.centroids[centroid_y][centroid_idx]
        del self.buffers[centroid_y][centroid_idx]
        del self.centroids_window_counts[centroid_y][centroid_idx]
        del self.centroids_window_buffers[centroid_y][centroid_idx]
        del self.centroids_window_last_update[centroid_y][centroid_idx]
        self.total_num_centroids -= 1

    def __check_centroids(self):
        centroids = copy.deepcopy(self.centroids)

        for cls, class_centroids in centroids.items():
            for centroid_idx, centroid in class_centroids.items():
                if self.t - self.centroids_window_last_update[cls][centroid_idx] >= self.split_period:
                    if sum(self.centroids_window_counts[cls][centroid_idx].values()) < 0.4 * self.window_size:
                        self.__remove_centroid(cls, centroid_idx)
                    else:
                        centroid_split, sec_cls = self.__check_centroid_split(cls, centroid_idx)

                        if centroid_split:
                            self.splits_num += 1
                            self.__extract_new_centroid(cls, centroid_idx, cls, split=True)
                            self.__extract_new_centroid(cls, centroid_idx, sec_cls, split=True)
                            self.__remove_centroid(cls, centroid_idx)

                        self.centroids_window_last_update[cls][centroid_idx] = self.t

    def __check_centroid_split(self, centroid_y, centroid_idx):
        counts = self.centroids_window_counts[centroid_y][centroid_idx]
        if len(counts) < 2:
            return False, -1

        [(first_cls, first_cnt), (sec_cls, sec_cnt)] = counts.most_common(2)
        if centroid_y != first_cls or sec_cnt == 0:
            return False, -1

        return (first_cnt / sec_cnt) - 1.0 <= self.split_thresh, sec_cls

    def sample(self, x_batch, y_batch, weights, **kwargs):
        num_samples_per_instance = self.total_num_centroids
        num_samples = num_samples_per_instance * len(x_batch)

        input_shape = x_batch[0].shape
        sampled_x_batch = np.zeros((num_samples, *input_shape))
        sampled_y_batch = np.zeros(num_samples)
        sampled_weights = np.zeros(num_samples)

        cls_indices = collections.defaultdict(list)
        i = 0

        for _ in range(len(x_batch)):
            for class_idx, centroid_buffers in self.buffers.items():
                for buffer_idx, centroid_buffer in centroid_buffers.items():
                    if self.__try_sample(self.centroids_window_counts[class_idx][buffer_idx], class_idx):
                        (rx, ry, rw) = random.choice(centroid_buffer)
                        sampled_x_batch[i, :] = rx[:]
                        sampled_y_batch[i] = ry
                        sampled_weights[i] = rw
                        cls_indices[ry].append(i)

                        i += 1

        return self.__resample(sampled_x_batch[:i], sampled_y_batch[:i], sampled_weights[:i], cls_indices)

    @staticmethod
    def __try_sample(counts, cls):
        if len(counts) == 1 and list(counts.keys())[0] == cls:
            return True
        else:
            [(first_cls, first_cnt), (sec_cls, sec_cnt)] = counts.most_common(2)
            if first_cls == cls:
                r = math.tanh(4 * (first_cnt - sec_cnt) / (first_cnt + sec_cnt))
                return r > random.random()
        return False

    @staticmethod
    def __resample(x_batch, y_batch, weights, cls_indices):
        max_cnt = max([len(indices) for indices in cls_indices.values()])
        num_samples = len(cls_indices) * max_cnt

        input_shape = x_batch[0].shape
        resampled_x_batch = np.zeros((num_samples, *input_shape))
        resampled_y_batch = np.zeros(num_samples)
        resampled_weights = np.zeros(num_samples)

        i = 0
        for cls, indices in cls_indices.items():
            while len(indices) < max_cnt:
                indices.append(random.choice(indices))

            resampled_x_batch[i:i + max_cnt, :] = x_batch[indices, :]
            resampled_y_batch[i:i + max_cnt] = y_batch[indices]
            resampled_weights[i:i + max_cnt] = weights[indices]
            i += len(indices)

        return resampled_x_batch, resampled_y_batch, resampled_weights

    def region(self, x):
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        has_cluster_nearby = False
        for y, class_centroids in self.centroids.items():
            for centroid_idx, centroid in class_centroids.items():
                mean, w_sum, var = centroid
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

# DDCW
class DDCW:
    def __init__(self, e, c, beta, ltc, device):
        # e: maximum ensemble size, c: number of classes
        # beta: beta, ltc: life time coefficient
        # device: device when testing
        self.e = e
        self.c = c
        self.beta = beta
        self.ltc = ltc
        self.device = device
        self.W = []
        self.ensemble = []

    def test(self, test_loader):
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            correct = 0
            total_loss = 0
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = torch.zeros((data.shape[0], self.c)).to(self.device)
                weight_vote = torch.zeros((data.shape[0], self.c)).to(self.device)
                for i in range(len(self.ensemble)):
                    classifier = self.ensemble[i]
                    classifier.to(self.device)
                    classifier.eval()
                    out = classifier(data)
                    # for calculating loss
                    output.add_(out * torch.tensor(self.W[i]).to(self.device))
                    # for prediction
                    _, pred = out.max(1)
                    vote = nn.functional.one_hot(pred, num_classes=self.c)
                    weight_vote.add_(vote * torch.tensor(self.W[i]).to(self.device))
                loss = criterion(output, target)
                total_loss += loss.item() * len(data)
                _, pred = weight_vote.max(1)
                correct += pred.eq(target).sum().item()
        acc = correct / len(test_loader.dataset)
        total_loss /= len(test_loader.dataset)
        return total_loss, acc

    def add_classifier(self, classifier):
        classifier.cpu()
        classifier.eval()
        self.ensemble.append(classifier)
        self.W.append([1. for i in range(self.c)])
        if len(self.ensemble) > self.e:
            self.ensemble.pop(0)
            self.W.pop(0)

    def update_accuracy(self, test_loader):
        for i in range(len(self.ensemble)):
            classifier = self.ensemble[i]
            classifier.to(self.device)
            classifier.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = classifier(data)
                    for E_pred, y in zip(output.argmax(1), target):
                        if E_pred == y:
                            self.W[i][y] *= self.beta

    def multiply_coeff(self):
        for i in range(len(self.W)-1): # exclude latest model
            for j in range(len(self.W[i])):
                self.W[i][j] *= self.ltc

    def update_diversity(self, test_loader, epsilon=1e-5):
        cnt = 0
        Q = 0
        for i in range(len(self.ensemble)):
            for j in range(i+1, len(self.ensemble)):
                # pair prediction
                classifier1 = self.ensemble[i]
                classifier1.to(self.device)
                classifier1.eval()
                classifier2 = self.ensemble[j]
                classifier2.to(self.device)
                classifier2.eval()
                N11, N00, N01, N10 = 0, 0, 0, 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output1 = classifier1(data)
                        output2 = classifier2(data)
                        for E1_pred, E2_pred, y in zip(output1.argmax(1), output2.argmax(1), target):
                            if E1_pred == y and E2_pred == y:
                                N11 += 1
                            elif E1_pred != y and E2_pred != y:
                                N00 += 1
                            elif E1_pred != y and E2_pred == y:
                                N01 += 1
                            else: # E1_pred == y and E2_pred != y:
                                N10 += 1
                Q  += (N11 * N00 - N01 * N10) / (N11 * N00 + N01 * N10 + epsilon) # add epsilon to denominator to avoid divided by zero
                cnt += 1
        Q /= cnt
        for i in range(len(self.W)):
            for j in range(len(self.W[i])):
                self.W[i][j] += (1 - Q)

    def normalize_weight(self):
        for i in range(len(self.W[0])):
            total_weight = 0
            for j in range(len(self.W)):
                total_weight += self.W[j][i]
            for j in range(len(self.W)):
                self.W[j][i] /= total_weight

# DDG-DA
class PredNet(nn.Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.W = nn.Parameter(torch.zeros(self.chunk_size, 1))
        self.W = nn.init.kaiming_normal_(self.W)

    def forward(self, X, y, X_test):
        assert X.shape[0] == self.chunk_size
        X_w = X.T * self.W.view(1, X.shape[0])
        # X_w = X.T * torch.sigmoid(self.W.T)
        theta = torch.inverse(X_w @ X) @ X_w @ y
        return torch.sigmoid(X_test @ theta)
        # return X_test @ theta
