import torch
import math
import torch.nn as nn

def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()

class  Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='none', bn=False, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        if activation == 'tanh':
            self.ac = nn.Tanh()
        elif activation == 'softmax':
            self.ac = nn.Softmax()
        elif activation == 'sigmoid':
            self.ac = nn.Sigmoid()
        elif activation == 'relu':
            self.ac = nn.ReLU() 
        else:
            self.ac = nn.Identity()
        
        if bn:
            self.bn = nn.BatchNorm1d(out_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        else:
            self.bn = nn.Identity()

        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(1. / out_features))

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.ac(x)
        return x

class AdditiveAttention(nn.Module):
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout=0):
        super(AdditiveAttention, self).__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        # self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        attention_weights = self.softmax(scores)
        return attention_weights.squeeze(1) * values.squeeze(-1)