import os
import pickle
import pandas as pd
from dataset import SAW2, RawSAW2
from tqdm import tqdm
from torch.utils.data import DataLoader
from helpers import *
from model import MTL, AMTL
import torch.optim as optim
import argparse

DATA_DIR = '../../../Data/ABAW4/'

def val(net, validldr):
    net.eval()
    all_y_va = None
    all_y_ex = None
    all_y_au = None
    all_mask_va = None
    all_mask_ex = None
    all_mask_au = None
    all_yhat_va = None
    all_yhat_ex = None
    all_yhat_au = None
    for batch_idx, (inputs, y_va, y_ex, y_au, mask_va, mask_ex, mask_au) in enumerate(tqdm(validldr)):
        with torch.no_grad():
            y_va = y_va.float()
            y_au = y_au.float()
            y_ex = y_ex.long()
            mask_va = mask_va.float()
            mask_ex = mask_ex.float()
            mask_au = mask_au.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                y_va = y_va.cuda()
                y_ex = y_ex.cuda()
                y_au = y_au.cuda()
                mask_va = mask_va.cuda()
                mask_ex = mask_ex.cuda()
                mask_au = mask_au.cuda()
            yhat_va, yhat_ex, yhat_au = net(inputs)

            if all_y_va == None:
                all_y_va = y_va.clone()
                all_y_ex = y_ex.clone()
                all_y_au = y_au.clone()
                all_mask_va = mask_va.clone()
                all_mask_ex = mask_ex.clone()
                all_mask_au = mask_au.clone()
                all_yhat_va = yhat_va.clone()
                all_yhat_ex = yhat_ex.clone()
                all_yhat_au = yhat_au.clone()
            else:
                all_y_va = torch.cat((all_y_va, y_va), 0)
                all_y_ex = torch.cat((all_y_ex, y_ex), 0)
                all_y_au = torch.cat((all_y_au, y_au), 0)
                all_mask_va = torch.cat((all_mask_va, mask_va), 0)
                all_mask_ex = torch.cat((all_mask_ex, mask_ex), 0)
                all_mask_au = torch.cat((all_mask_au, mask_au), 0)
                all_yhat_va = torch.cat((all_yhat_va, yhat_va), 0)
                all_yhat_ex = torch.cat((all_yhat_ex, yhat_ex), 0)
                all_yhat_au = torch.cat((all_yhat_au, yhat_au), 0)
    all_y_va = all_y_va.cpu().numpy()
    all_y_ex = all_y_ex.cpu().numpy()
    all_y_au = all_y_au.cpu().numpy()
    all_mask_va = all_mask_va.cpu().numpy()
    all_mask_ex = all_mask_ex.cpu().numpy()
    all_mask_au = all_mask_au.cpu().numpy()
    all_yhat_va = all_yhat_va.cpu().numpy()
    all_yhat_ex = all_yhat_ex.cpu().numpy()
    all_yhat_au = all_yhat_au.cpu().numpy()
    all_y_va = all_y_va[all_mask_va == 1]
    all_y_ex = all_y_ex[all_mask_ex == 1]
    all_y_au = all_y_au[all_mask_au == 1]
    all_yhat_va = all_yhat_va[all_mask_va == 1]
    all_yhat_ex = all_yhat_ex[all_mask_ex == 1]
    all_yhat_au = all_yhat_au[all_mask_au == 1]
    va_metrics = VA_metric(all_y_va, all_yhat_va)
    ex_metrics = EX_metric(all_y_ex, all_yhat_ex)
    au_metrics = AU_metric(all_y_au, all_yhat_au)
    performance = va_metrics + ex_metrics + au_metrics
    return va_metrics, ex_metrics, au_metrics, performance


def main():
    parser = argparse.ArgumentParser(description='Validate task')
    parser.add_argument('--input', '-i', default='', help='Input file')
    parser.add_argument('--net', '-n', default='amtl', help='Net name')
    parser.add_argument('--batch', '-b', type=int, default=256, help='Batch size')
    args = parser.parse_args()
    resume = args.input
    net_name = args.net
    batch_size = args.batch

    valid_file = os.path.join(DATA_DIR, 'validation_set_annotations.txt')
    image_path = os.path.join(DATA_DIR,'cropped_aligned')
    with open(os.path.join(DATA_DIR, 'saw2_enet_b0_8_best_vgaf.pickle'), 'rb') as handle:
        filename2featuresAll=pickle.load(handle)

    validset = SAW2(valid_file, filename2featuresAll)
    validset = RawSAW2(valid_file, image_path)
    validldr = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)

    if net_name == 'amtl':
        net = AMTL(in_channels=1288, extractor='./model/enet_b0_8_best_vgaf.pt')
    else:
        net = MTL(in_channels=1288, extractor='./model/enet_b0_8_best_vgaf.pt')

    if resume != '':
        print("Resume form | {} ]".format(resume))
        net = load_state_dict(net, resume)

    net = nn.DataParallel(net).cuda()

    val_metrics_va, val_metrics_ex, val_metrics_au, val_metrics = val(net, validldr)

    infostr = {'EpochNo: {:.5f},{:.5f},{:.5f},{:.5f}'
            .format(val_metrics_va,
                    val_metrics_ex,
                    val_metrics_au,
                    val_metrics)}
    print(infostr)

if __name__=="__main__":
    main()

