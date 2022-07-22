import enum
import os
import pickle
import pandas as pd
from dataset import SAW2, ImageSAW2
from tqdm import tqdm
from torch.utils.data import DataLoader
from helpers import *
from model import MTL, AMTL
import torch.optim as optim
import argparse

DATA_DIR = '../../../Data/ABAW4/'

def test(net, testldr):
    net.eval()
    all_yhat_va = None
    all_yhat_ex = None
    all_yhat_au = None
    for batch_idx, inputs in enumerate(tqdm(testldr)):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            yhat_va, yhat_ex, yhat_au = net(inputs)

            if all_yhat_va == None:
                all_yhat_va = yhat_va.clone()
                all_yhat_ex = yhat_ex.clone()
                all_yhat_au = yhat_au.clone()
            else:
                all_yhat_va = torch.cat((all_yhat_va, yhat_va), 0)
                all_yhat_ex = torch.cat((all_yhat_ex, yhat_ex), 0)
                all_yhat_au = torch.cat((all_yhat_au, yhat_au), 0)
    all_yhat_va = all_yhat_va.cpu().numpy()
    all_yhat_ex = all_yhat_ex.cpu().numpy()
    all_yhat_au = all_yhat_au.cpu().numpy()
    return all_yhat_va, all_yhat_ex, all_yhat_au

def generate_output(filename, img, va, ex, au):
    with open(os.path.join(DATA_DIR, 'testset', filename+'.txt'), 'w') as f:
        print(va.shape, ex.shape, au.shape)
        f.write("image,valence,arousal,expression,aus\n")

        for i, name in enumerate(img):
            a = (au[i,:] >= 0.5) * 1
            infostr = '{},{:.9f},{:.9f},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(name,
                            va[i,0], va[i,1],
                            np.argmax(ex[i,:]),
                            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11])
            f.write(infostr)

def main():
    parser = argparse.ArgumentParser(description='Validate task')
    parser.add_argument('--input', '-i', default='', help='Input file')
    parser.add_argument('--net', '-n', default='mefarg', help='Net name')
    parser.add_argument('--batch', '-b', type=int, default=256, help='Batch size')
    args = parser.parse_args()
    resume = args.input
    net_name = args.net
    batch_size = args.batch

    test_file = os.path.join(DATA_DIR, 'testset', 'MTL_Challenge_test_set_release.txt')
    image_path = os.path.join(DATA_DIR, 'testset', 'cropped_aligned')

    testset = ImageSAW2(test_file, image_path)
    testldr = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    if net_name == 'mefarg':
        net = MTL(in_channels=1288, extractor='./model/enet_b0_8_best_vgaf.pt')
    else:
        net = AMTL(in_channels=1288, extractor='./model/enet_b0_8_best_vgaf.pt')

    if resume != '':
        print("Resume form | {} ]".format(resume))
        net = load_state_dict(net, resume)

    net = nn.DataParallel(net).cuda()

    va, ex, au = test(net, testldr)

    generate_output(args.input, testset.X, va, ex, au)


if __name__=="__main__":
    main()

