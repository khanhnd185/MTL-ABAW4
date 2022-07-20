import enum
import os
import pickle
import pandas as pd
from dataset import SAW2, ImageSAW2
from tqdm import tqdm
from torch.utils.data import DataLoader
from helpers import *
from model import MEFARG, AMEFARG
import torch.optim as optim
import argparse


batch_size = 256
num_workers = 0
epochs = 20
DATA_DIR = '../../../Data/ABAW4/'
learning_rate = 1e-3
resume = ''
output_dir = './results'
early_stop = None

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
    parser.add_argument('--input', '-i', default='2.pth', help='Input file')
    parser.add_argument('--net', '-n', default='mefarg', help='Net name')
    args = parser.parse_args()
    resume = args.input
    net_name = args.net

    test_file = os.path.join(DATA_DIR, 'testset', 'MTL_Challenge_test_set_release.txt')
    image_path = os.path.join(DATA_DIR, 'testset', 'cropped_aligned')
    with open(os.path.join(DATA_DIR, 'enet0_8.pickle'), 'rb') as handle:
        filename2featuresAll=pickle.load(handle)

    testset = ImageSAW2(test_file, image_path)
    testldr = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if net_name == 'mefarg':
        net = MEFARG(in_channels=1288, e2e=True)
    else:
        net = AMEFARG(in_channels=1288)

    if resume != '':
        print("Resume form | {} ]".format(resume))
        net = load_state_dict(net, resume)

    net = nn.DataParallel(net).cuda()

    va, ex, au = test(net, testldr)

    generate_output(args.input, testset.X, va, ex, au)


if __name__=="__main__":
    main()

