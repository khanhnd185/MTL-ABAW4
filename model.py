import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from helpers import normalize_digraph
from block import *


class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels,self.in_channels)
        self.V = nn.Linear(self.in_channels,self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == 'dots':
            si = x.detach()
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'cosine':
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'l1':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x


class ANFL(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(ANFL, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = Dense(self.in_channels, self.in_channels, bn=True, drop=0.5, activation='relu')
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.gnn = GNN(self.in_channels, self.num_classes,neighbor_num=neighbor_num,metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        # f_v = f_u.mean(dim=-2)
        f_v = f_u
        # FGG
        f_v = self.gnn(f_v)
        b, n, c = f_v.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        return cl, f_v

class Extractor(nn.Module):
    def __init__(self, path):
        super(Extractor, self).__init__()
        self.backbone = torch.load(path)
        self.classifier = self.backbone.classifier[0]
        self.backbone.classifier  = torch.nn.Identity()
    
    def forward(self, x):
        x = self.backbone(x)
        score = self.classifier(x)
        x = torch.cat((x, score), dim=-1)
        return x

class MTL(nn.Module):
    def __init__(self, in_channels=1288, neighbor_num=4, metric='dots', extractor=None, e2e=False):
        super(MTL, self).__init__()
        self.AU_metric_dim = 12
        self.EX_metric_dim = 8
        self.VA_metric_dim = 2

        if extractor is not None:
            self.extractor = Extractor(extractor)
        else:
            self.extractor = nn.Identity()

        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.global_linear = Dense(self.in_channels, self.out_channels, bn=True, drop=0.5)
        self.au = ANFL(self.out_channels, self.AU_metric_dim, neighbor_num, metric)

        self.va = Dense(in_channels, self.VA_metric_dim, activation='tanh', bn=True)
        self.ex = Dense(in_channels, self.EX_metric_dim, activation='none')

        for p in self.extractor.parameters():
            p.requires_grad = False
    def forward(self, x):
        x = self.extractor(x)
        au, f_v = self.au(x)
        va = self.va(x)
        ex = self.ex(x)

        return va, ex, au

class AMTL(nn.Module):
    def __init__(self, in_channels=1288, neighbor_num=4, metric='dots', extractor=None, freeze_au=True):
        super(AMTL, self).__init__()
        self.AU_metric_dim = 12
        self.EX_metric_dim = 8
        self.VA_metric_dim = 2
        self.attention_dim = 64

        if extractor is not None:
            self.extractor = Extractor(extractor)
        else:
            self.extractor = nn.Identity()

        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.au = ANFL(self.out_channels, self.AU_metric_dim, neighbor_num, metric)

        self.va = Dense(in_channels, self.VA_metric_dim, activation='tanh', bn=True)
        self.ex = Dense(in_channels, self.EX_metric_dim, activation='none')

        self.attention = AdditiveAttention(1, self.AU_metric_dim, self.attention_dim)

        for p in self.extractor.parameters():
            p.requires_grad = False

        if freeze_au:
            for p in self.au.parameters():
                p.requires_grad = False

            
    def forward(self, x):
        x = self.extractor(x)
        au, f_v = self.au(x)
        va = self.va(x)
        ex = self.ex(x)

        q = au.unsqueeze(1)
        v = ex.unsqueeze(-1)
        ex = self.attention(q, v, v)

        return va, ex, au
