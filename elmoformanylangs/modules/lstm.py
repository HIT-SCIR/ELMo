from __future__ import absolute_import
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy


class LstmbiLm(nn.Module):
  def __init__(self, config, use_cuda=False):
    super(LstmbiLm, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    
    self.encoder = nn.LSTM(self.config['encoder']['projection_dim'],
                           self.config['encoder']['dim'],
                           num_layers=self.config['encoder']['n_layers'], 
                           bidirectional=True,
                           batch_first=True, 
                           dropout=self.config['dropout'])
    self.projection = nn.Linear(self.config['encoder']['dim'], self.config['encoder']['projection_dim'], bias=True)

  def forward(self, inputs):
    forward, backward = self.encoder(inputs)[0].split(self.config['encoder']['dim'], 2)
    return torch.cat([self.projection(forward), self.projection(backward)], dim=2)
