import numpy as np
import pandas as pd
import copy
import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt

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
    