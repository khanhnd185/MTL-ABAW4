from copy import deepcopy
import torch
import torch.nn as nn
from helpers import *
from model import MEFARG

net = MEFARG(in_channels=1288)
net = load_state_dict(net, '1.pth')
# net_va = MEFARG(in_channels=1288)
# net_va = load_state_dict(net_va, 'results/train_mefarg_uni_VA_sam/best_val_perform.pth')
net_au = MEFARG(in_channels=1288)
# net_au = load_state_dict(net_au, 'results/mefarg_uni_AU_sam/80_epoch.pth') # best val loss
# net_au = load_state_dict(net_au, 'results/mefarg_uni_AU_sam/99_epoch.pth') # best train loss
net_au = load_state_dict(net_au, '../MTL-ABAW3/results/mefarg_uni_AU_sam/47_epoch.pth') # best val perform
net.head = deepcopy(net_au.head)
c = {'state_dict':net.state_dict()}
torch.save(c, '2.pth')
