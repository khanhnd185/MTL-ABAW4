import os
import pickle
import pandas as pd
from dataset import UniSAW2
from tqdm import tqdm
from torch.utils.data import DataLoader
from helpers import *
from model import MTL, AMTL
import torch.optim as optim
import argparse
from sam import SAM

DATA_DIR = '../../../Data/ABAW4/'
metric_func = {
    'VA': VA_metric,
    'EX': EX_metric,
    'AU': AU_metric,
}

def train(net, trainldr, optimizer, epoch, criteria, task):
    total_losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)
    yhat = {}
    for batch_idx, (inputs, y) in enumerate(tqdm(trainldr)):
        mask = torch.ones(inputs.shape[0])
        mask = mask.float()
        mask = mask.cuda()
        # adjust_learning_rate(optimizer, epoch, epochs, learning_rate, batch_idx, train_loader_len)
        if task == 'EX':
            y = y.long()
        else:
            y = y.float()
        inputs = inputs.cuda()
        y = y.cuda()

        yhat['VA'], yhat['EX'], yhat['AU'] = net(inputs)
        loss = criteria[task](yhat[task], y, mask)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        yhat['VA'], yhat['EX'], yhat['AU'] = net(inputs)
        criteria[task](yhat[task], y, mask).backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)

        total_losses.update(loss.data.item(), inputs.size(0))
    return total_losses.avg()


def val(net, validldr, criteria, task):
    total_losses = AverageMeter()
    yhat = {}
    net.eval()
    all_y = None
    all_yhat = None
    for batch_idx, (inputs, y) in enumerate(tqdm(validldr)):
        mask = torch.ones(inputs.shape[0])
        mask = mask.float()
        mask = mask.cuda()
        with torch.no_grad():
            if task == 'EX':
                y = y.long()
            else:
                y = y.float()
            inputs = inputs.cuda()
            y = y.cuda()
            yhat['VA'], yhat['EX'], yhat['AU'] = net(inputs)
            loss = criteria[task](yhat[task], y, mask)
            total_losses.update(loss.data.item(), inputs.size(0))

            if all_y == None:
                all_y = y.clone()
                all_yhat = yhat[task].clone()
            else:
                all_y = torch.cat((all_y, y), 0)
                all_yhat = torch.cat((all_yhat, yhat[task]), 0)
    all_y = all_y.cpu().numpy()
    all_yhat = all_yhat.cpu().numpy()
    metrics = metric_func[task](all_y, all_yhat)
    return total_losses.avg(), metrics


def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--net', '-n', default='amtl', help='Net name')
    parser.add_argument('--input', '-i', default='', help='Input file')
    parser.add_argument('--task', '-t', default='AU', help='Task')
    parser.add_argument('--batch', '-b', type=int, default=256, help='Batch size')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of epoches')
    parser.add_argument('--lr', '-a', type=float, default=1e-3, help='Learning rate')

    args = parser.parse_args()
    task = args.task
    resume = args.input
    net_name = args.net
    batch_size = args.batch
    epochs = args.epoch
    learning_rate = args.lr
    output_dir = 'train_' + net_name + '_uni_' + task 

    train_file = os.path.join(DATA_DIR, 'training_set_annotations.txt')
    valid_file = os.path.join(DATA_DIR, 'validation_set_annotations.txt')
    image_path = os.path.join(DATA_DIR,'cropped_aligned')

    with open(os.path.join(DATA_DIR, 'saw2_enet_b0_8_best_vgaf_aug.pickle'), 'rb') as handle:
        filename2featuresAll=pickle.load(handle)
    with open(os.path.join(DATA_DIR, 'saw2_enet_b0_8_best_vgaf.pickle'), 'rb') as handle:
        filename2featuresAll_val=pickle.load(handle)

    trainset = UniSAW2(train_file, filename2featuresAll, task)
    validset = UniSAW2(valid_file, filename2featuresAll_val, task)

    trainldr = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    validldr = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)

    start_epoch = 0
    if net_name == 'amtl':
        net = AMTL(in_channels=1288)
    else:
        net = MTL(in_channels=1288)

    train_ex_weight = torch.from_numpy(trainset.ex_weight())
    valid_ex_weight = torch.from_numpy(validset.ex_weight())
    train_au_weight = torch.from_numpy(trainset.au_weight())
    valid_au_weight = torch.from_numpy(validset.au_weight())
    train_ex_weight = train_ex_weight.float()
    valid_ex_weight = valid_ex_weight.float()
    train_au_weight = train_au_weight.float()
    valid_au_weight = valid_au_weight.float()


    if resume != '':
        print("Resume form | {} ]".format(resume))
        net = load_state_dict(net, resume)

    net = nn.DataParallel(net).cuda()
    train_au_weight = train_au_weight.cuda()
    train_ex_weight = train_ex_weight.cuda()
    valid_au_weight = valid_au_weight.cuda()
    valid_ex_weight = valid_ex_weight.cuda()

    train_criteria = {}
    valid_criteria = {}
    train_criteria['VA'] = RegressionLoss()
    valid_criteria['VA'] = RegressionLoss()
    train_criteria['AU'] = WeightedAsymmetricLoss(weight=train_au_weight)
    valid_criteria['AU'] = WeightedAsymmetricLoss(weight=valid_au_weight)
    train_criteria['EX'] = MaskedCELoss(weight=train_ex_weight, ignore_index=-1)
    valid_criteria['EX'] = MaskedCELoss(weight=valid_ex_weight, ignore_index=-1)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(net.parameters(), base_optimizer, lr=learning_rate, momentum=0.9, weight_decay=1.0/batch_size)
    best_performance = 0.0
    epoch_from_last_improvement = 0

    df = {}
    df['epoch'] = []
    df['lr'] = []
    df['train_loss'] = []
    df['val_loss'] = []
    df['val_metrics'] = []

    for epoch in range(start_epoch, epochs):
        lr = optimizer.param_groups[0]['lr']
        train_loss = train(net, trainldr, optimizer, epoch, train_criteria, task)
        val_loss, val_metrics = val(net, validldr, valid_criteria, task)

        infostr = {'Task {}: {},{:.5f},{:.5f},{:.5f},{:.5f}'
                .format(task,
                        epoch,
                        lr,
                        train_loss,
                        val_loss,
                        val_metrics)}
        print(infostr)

        os.makedirs(os.path.join('results', output_dir), exist_ok = True)

        if val_metrics >= best_performance:
            checkpoint = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join('results', output_dir, 'best_val_perform.pth'))
            best_performance = val_metrics
            epoch_from_last_improvement = 0
        else:
            epoch_from_last_improvement += 1

        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join('results', output_dir, 'cur_model.pth'))

        df['epoch'].append(epoch)
        df['lr'].append(lr)
        df['train_loss'].append(train_loss)
        df['val_loss'].append(val_loss)
        df['val_metrics'].append(val_metrics)
   

    df = pd.DataFrame(df)
    csv_name = os.path.join('results', output_dir, 'train.csv')
    df.to_csv(csv_name)

if __name__=="__main__":
    main()

