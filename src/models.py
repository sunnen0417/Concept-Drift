import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np

import math

class Classifier(nn.Module):
    def __init__(self, in_size=2, out_size=2):
        super(Classifier, self).__init__()
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
