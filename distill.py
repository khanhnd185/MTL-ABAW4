import os
import pickle
import argparse
import pandas as pd
from sam import SAM
from tqdm import tqdm
from model import *
from helpers import *
from dataset import UniSAW2
from torch.utils.data import DataLoader

metric_func = {
    'VA': VA_metric,
    'EX': EX_metric,
    'AU': AU_metric,
}

def train(teacher, student, trainldr, optimizer, epoch, epochs, learning_rate, criteria, task):
    total_losses = AverageMeter()
    student.train()
    teacher.eval()
    train_loader_len = len(trainldr)
    yhat = {}
    ysoft = {}
    for batch_idx, (inputs, y) in enumerate(tqdm(trainldr)):
        mask = torch.ones(inputs.shape[0])
        mask = mask.float()
        mask = mask.cuda()
        adjust_learning_rate(optimizer, epoch, epochs, learning_rate, batch_idx, train_loader_len)
        if task == 'EX':
            y = y.long()
        else:
            y = y.float()
        inputs = inputs.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        yhat['VA'], yhat['EX'], yhat['AU'] = student(inputs)
        with torch.no_grad():
            ysoft['VA'], ysoft['EX'], ysoft['AU'] = teacher(inputs)
        loss = criteria[task](yhat[task], ysoft[task], y, mask)
        loss.backward()
        optimizer.step()

        total_losses.update(loss.data.item(), inputs.size(0))
    return total_losses.avg()

def train_sam(teacher, student, trainldr, optimizer, epoch, epochs, learning_rate, criteria, task):
    total_losses = AverageMeter()
    student.train()
    teacher.eval()
    train_loader_len = len(trainldr)
    yhat = {}
    ysoft = {}
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

        yhat['VA'], yhat['EX'], yhat['AU'] = student(inputs)
        with torch.no_grad():
            ysoft['VA'], ysoft['EX'], ysoft['AU'] = teacher(inputs)
        loss = criteria[task](yhat[task], ysoft[task], y, mask)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        yhat['VA'], yhat['EX'], yhat['AU'] = student(inputs)
        criteria[task](yhat[task], ysoft[task], y, mask).backward()
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
    metrics, _ = metric_func[task](all_y, all_yhat)
    return total_losses.avg(), metrics


def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--net', '-n', default='amtl', help='Net name')
    parser.add_argument('--teacher', '-i', default='2b.pth', help='Teacher model')
    parser.add_argument('--task', '-t', default='EX', help='Task')
    parser.add_argument('--temp', '-T', type=int, default=1, help='Temperature')
    parser.add_argument('--batch', '-b', type=int, default=256, help='Batch size')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of epoches')
    parser.add_argument('--lr', '-r', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--alpha', '-a', type=float, default=1e-1, help='Distillation weight')
    parser.add_argument('--datadir', '-d', default='../../../Data/ABAW4/', help='Data folder path')
    parser.add_argument('--tfeature', '-f', default='saw2_enet_b0_8_best_vgaf_aug.pickle', help='Train image feature')
    parser.add_argument('--vfeature', '-v', default='saw2_enet_b0_8_best_vgaf.pickle', help='Validation image feature')
    parser.add_argument('--sam', '-s', action='store_true', help='Apply SAM optimizer')

    args = parser.parse_args()
    task = args.task
    temp = args.temp
    alpha = args.alpha
    epochs = args.epoch
    teacher_name = args.teacher
    net_name = args.net
    data_dir = args.datadir
    batch_size = args.batch
    learning_rate = args.lr
    train_feature = args.tfeature
    valid_feature = args.vfeature
    use_sam = (args.sam == True)
    output_dir = 'distill1_lr' + str(learning_rate) + '_a' + str(alpha) + '_task' + task

    train_file = os.path.join(data_dir, 'training_set_annotations.txt')
    valid_file = os.path.join(data_dir, 'validation_set_annotations.txt')

    with open(os.path.join(data_dir, train_feature), 'rb') as handle:
        train_feature_dict = pickle.load(handle)
    with open(os.path.join(data_dir, valid_feature), 'rb') as handle:
        valid_feature_dict = pickle.load(handle)

    trainset = UniSAW2(train_file, train_feature_dict, task)
    validset = UniSAW2(valid_file, valid_feature_dict, task)

    trainldr = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    validldr = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)

    start_epoch = 0
    if net_name == 'amtl':
        teacher = AMTL(freeze_au=(task != 'AU'))
        student = AMTL(freeze_au=(task != 'AU'))
    elif net_name == 'fusion':
        teacher = FMTL(freeze_au=(task != 'AU'))
        student = FMTL(freeze_au=(task != 'AU'))
    elif net_name == 'multihead':
        teacher = MultiHead(1288, freeze_au=(task != 'AU'))
        student = MultiHead(1288, freeze_au=(task != 'AU'))
    else:
        teacher = MTL(freeze_au=(task != 'AU'))
        student = MTL(freeze_au=(task != 'AU'))

    train_ex_weight = torch.from_numpy(trainset.ex_weight())
    valid_ex_weight = torch.from_numpy(validset.ex_weight())
    train_au_weight = torch.from_numpy(trainset.au_weight())
    valid_au_weight = torch.from_numpy(validset.au_weight())
    train_ex_weight = train_ex_weight.float()
    valid_ex_weight = valid_ex_weight.float()
    train_au_weight = train_au_weight.float()
    valid_au_weight = valid_au_weight.float()


    print("Teacher model | {} ]".format(teacher_name))
    teacher = load_state_dict(teacher, teacher_name)

    teacher = nn.DataParallel(teacher).cuda()
    student = nn.DataParallel(student).cuda()
    train_au_weight = train_au_weight.cuda()
    train_ex_weight = train_ex_weight.cuda()
    valid_au_weight = valid_au_weight.cuda()
    valid_ex_weight = valid_ex_weight.cuda()

    train_criteria = {}
    valid_criteria = {}
    train_criteria['VA'] = DistillationLoss(alpha, RegressionLoss())
    train_criteria['AU'] = DistillationLoss(alpha, WeightedAsymmetricLoss(weight=train_au_weight))
    train_criteria['EX'] = DistillationLossFromLogit(alpha, MaskedCELoss(weight=train_ex_weight, ignore_index=-1), temp)
    valid_criteria['VA'] = RegressionLoss()
    valid_criteria['AU'] = WeightedAsymmetricLoss(weight=valid_au_weight)
    valid_criteria['EX'] = MaskedCELoss(weight=valid_ex_weight, ignore_index=-1)

    if use_sam:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(student.parameters(), base_optimizer, lr=learning_rate, momentum=0.9, weight_decay=1.0/batch_size)
    else:
        optimizer = torch.optim.AdamW(student.parameters(), betas=(0.9, 0.999), lr=learning_rate, weight_decay=1.0/batch_size)
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
        if use_sam:
            train_loss = train_sam(teacher, student, trainldr, optimizer, epoch, epochs, learning_rate, train_criteria, task)
        else:
            train_loss = train(teacher, student, trainldr, optimizer, epoch, epochs, learning_rate, train_criteria, task)
        val_loss, val_metrics = val(student, validldr, valid_criteria, task)

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
                'state_dict': student.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join('results', output_dir, 'best_val_perform.pth'))
            best_performance = val_metrics
            epoch_from_last_improvement = 0
        else:
            epoch_from_last_improvement += 1

        checkpoint = {
            'epoch': epoch,
            'state_dict': student.state_dict(),
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

