import os
import shutil
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from Datasets import RotStnDataset
# from utils import stn_utils

best_losses = 1


def main_worker(args):
    global best_losses

    # 1. create model
    model = models.__dict__[args.arch]()

    # trained_path = 'trained_models/la/stn_net/0722_stn2_8classes_L2_3+0.1/224_lr2e_5_28'
    # params_dict = os.path.join(trained_path, 'epoch-{}.pt'.format(args.start_epoch))
    # model.load_state_dict(torch.load(params_dict))

    model = model.cuda()

    # 2. define loss function (criterion) and optimizer
    criterion = (nn.L1Loss().cuda(), nn.MSELoss().cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cudnn.benchmark = True

    # 3. Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.dataset == 'Stn4ClassesDataset':
        folder_train_dataset = datasets.ImageFolder(root=traindir, transform=transforms.Compose([
            transforms.Resize(224), transforms.ToTensor(), normalize]))
        folder_val_dataset = datasets.ImageFolder(root=valdir, transform=transforms.Compose([
            transforms.Resize(224), transforms.ToTensor(), normalize]))

        train_dataset = RotStnDataset.Stn4ClassesDatasetDiffYear(folder_train_dataset)
        val_dataset = RotStnDataset.Stn4ClassesDatasetDiffYear(folder_val_dataset)

    elif args.dataset == 'StnNClassesDatasetDiffYear':
        folder_train_dataset = datasets.ImageFolder(root=traindir, transform=transforms.Compose([
            transforms.Resize(370), transforms.ToTensor(), normalize]))
        folder_val_dataset = datasets.ImageFolder(root=valdir, transform=transforms.Compose([
            transforms.Resize(370), transforms.ToTensor(), normalize]))

        train_dataset = RotStnDataset.StnNClassesDatasetDiffYear(folder_train_dataset)
        val_dataset = RotStnDataset.StnNClassesDatasetDiffYear(folder_val_dataset)
    else:
        raise ValueError('Wrong dataset class')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=False, sampler=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=False)

    writer = SummaryWriter()
    epoch_counter = []
    train_epoch_loss = []
    val_epoch_loss = []
    min_theta_loss = 1

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # TRAIN for one epoch
        train_loss, train_losses_theta, train_losses_v = train(train_loader, model, criterion, optimizer, epoch, args)
        writer.add_scalars('Loss/group', {'train_loss': train_loss}, epoch+1)
        writer.add_scalars('Loss/group', {'train_loss_theta': train_losses_theta}, epoch + 1)
        writer.add_scalars('Loss/group', {'train_losses_v': train_losses_v}, epoch + 1)

        # EVALUATE on validation set
        test_losses_avg, test_losses_theta, test_losses_v = validate(val_loader, model, criterion, args)
        writer.add_scalars('Loss/group', {'test_loss': test_losses_avg}, epoch+1)
        writer.add_scalars('Loss/group', {'test_loss_theta': test_losses_theta}, epoch+1)
        writer.add_scalars('Loss/group', {'test_losses_v': test_losses_v}, epoch+1)

        # update plot params
        epoch_counter.append(epoch+1)
        train_epoch_loss.append(train_loss)
        val_epoch_loss.append(test_losses_avg)

        if test_losses_theta < min_theta_loss:
            torch.save(model.state_dict(), os.path.join(args.trained_path, 'epoch-{}.pt'.format(epoch + 1)))
            min_theta_loss = test_losses_theta

        if (epoch+1) % 20 == 0:
            show_epoch_plot(epoch, epoch_counter, train_epoch_loss, val_epoch_loss, args.trained_path)
            torch.save(model.state_dict(), os.path.join(args.trained_path, 'epoch-{}.pt'.format(epoch + 1)))


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    losses_theta = AverageMeter('Loss', ':.4f')
    losses_v = AverageMeter('Loss', ':.4f')

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), losses, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    for i, ((input0, theta0), (input1, theta1)) in enumerate(train_loader):
        # if args.gpu is not None:
        input0 = input0.cuda()
        input1 = input1.cuda()
        input_theta0 = theta0.float().cuda()
        input_theta1 = theta1.float().cuda()

        # compute output
        # output0, output1 = model(input0, input_theta0, input1, input_theta1)
        output0, output1 = model(input0, input_theta0, input1, input_theta1)

        feat0, theta0 = output0
        feat1, theta1 = output1

        loss_feats = criterion[0](feat0, feat1)
        loss_theta = criterion[1](input_theta0 + theta0.view(-1), input_theta1 + theta1.view(-1))
        loss = args.lambda_v * loss_feats + loss_theta

        # measure accuracy and record loss
        losses.update(loss.item(), input0.size(0))
        losses_theta.update(loss_theta.item(), input0.size(0))
        losses_v.update(loss_feats.item(), input0.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.print(i)

    return losses.avg, losses_theta.avg, losses_v.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    losses_theta = AverageMeter('Loss', ':.4f')
    losses_v = AverageMeter('Loss', ':.4f')

    progress = ProgressMeter(len(val_loader), losses, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, ((input0, input_theta0), (input1, input_theta1)) in enumerate(val_loader):
            input0 = input0.cuda()
            input1 = input1.cuda()
            input_theta0 = input_theta0.float().cuda()
            input_theta1 = input_theta1.float().cuda()

            # compute output
            output0, output1 = model(input0, input_theta0, input1, input_theta1)
            feat0, theta0 = output0
            feat1, theta1 = output1

            theta0 = theta0.view(-1)
            theta1 = theta1.view(-1)

            loss_feats = criterion[0](feat0, feat1)
            loss_theta = criterion[1](input_theta0 + theta0, input_theta1 + theta1)
            loss = args.lambda_v * loss_feats + loss_theta

            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input0.size(0))
            losses_theta.update(loss_theta.item(), input0.size(0))
            losses_v.update(loss_feats.item(), input0.size(0))

            print('input_theta0', input_theta0)
            print('theta0:', theta0)
            print('theta1:', theta1)

        # TODO: this should also be done with the ProgressMeter
        print(' * Val losses avg {losses.avg:.4f}'.format(losses=losses))

    return losses.avg, losses_theta.avg, losses_v.avg


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.trained_path, './model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def show_iter_plot(iteration, loss):
    plt.plot(iteration[1:], loss[1:])
    plt.show()


def show_epoch_plot(epoch, plt_epoch, epoch_loss, plt_val_loss, path):
    fig = plt.figure()
    plt.plot(plt_epoch, epoch_loss)
    plt.plot(plt_epoch, plt_val_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('epoch-{}'.format(epoch+1))
    plt.legend(['train', 'validation'])
    plt.show()
    fig.savefig(os.path.join(path, 'epoch-{}.tiff'.format(epoch+1)))


