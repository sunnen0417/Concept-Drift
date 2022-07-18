import math
import torch
import torch.nn as nn

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
