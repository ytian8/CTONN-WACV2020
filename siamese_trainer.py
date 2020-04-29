import torch
import os
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import models
from utils.losses import ContrastiveLoss
from Datasets.siameseDataset import DataAugSiaDataset


def main_worker(args):
    # 1. create model
    model = models.__dict__[args.arch]()
    # trained_path = 'trained_models/sfo/siamese_net_rot102loss_8pairs/lr1e_5'
    # pretrained_path = os.path.join(trained_path, 'epoch-{}.pt'.format(36))
    # model.load_state_dict(torch.load(pretrained_path))
    model = model.cuda()

    # 2. define loss function (criterion) and optimizer
    criterion = ContrastiveLoss(margin=1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(
    #     [{'params': model.embedding_net.convnet.parameters(), 'lr': args.lr/10.},
    #      {'params': model.embedding_net.fc.parameters()}], lr=args.lr)
    optimizer = torch.optim.Adam(
        [{'params': model.embedding_net.convnet.parameters(), 'lr': args.lr/10.},
         # {'params': model.embedding_net.conv5.parameters(), 'lr': args.lr/10.},
         {'params': model.embedding_net.fc.parameters()}], lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=-1)

    cudnn.benchmark = True

    # 3. Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_dataset = DataAugSiaDataset(root=traindir)
    val_dataset = DataAugSiaDataset(root=valdir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    cuda = torch.cuda.is_available()
    fit(args, train_loader, val_loader, model, criterion, optimizer, scheduler, args.epochs, cuda, log_interval=100)


def fit(args, train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, start_epoch=0):
    """
    Siamese network: Siamese loader, siamese model, contrastive loss
    """

    counter = []
    loss_history = []
    iteration_number = 0
    plt_epoch = []
    epoch_loss = []
    plt_val_loss = []
    writer = SummaryWriter()
    min_loss = 1

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, train_conv_loss, counter, loss_history, iteration_number = train_epoch(
            train_loader, model, loss_fn, optimizer, cuda, log_interval, counter, loss_history, iteration_number)
        writer.add_scalars('Loss/group', {'train_loss': train_loss}, epoch + 1)
        writer.add_scalars('Loss/group', {'train__conv_loss': train_conv_loss}, epoch + 1)

        plt_epoch.append(epoch+1)
        epoch_loss.append(train_loss)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

        val_loss, conv_loss = test_epoch(val_loader, model, loss_fn, cuda)
        print(len(val_loader))
        val_loss /= len(val_loader)
        conv_loss /= len(val_loader)
        plt_val_loss.append(val_loss)
        writer.add_scalars('Loss/group', {'test_loss': val_loss}, epoch + 1)
        writer.add_scalars('Loss/group', {'test_conv_loss': conv_loss}, epoch + 1)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)

        print(message)
        # if (epoch+1) >= 50:
        #     torch.save(model.state_dict(), os.path.join(args.trained_path, 'epoch-{}.pt'.format(epoch + 1)))

        if val_loss < min_loss:
            torch.save(model.state_dict(), os.path.join(args.trained_path, 'epoch-{}.pt'.format(epoch + 1)))
            min_loss = val_loss

        if (epoch+1) % 20 == 0:
            show_epoch_plot(epoch, plt_epoch, epoch_loss, plt_val_loss, args.trained_path)

        # if (epoch+1) % 5 == 0:
            # for param_group in optimizer.param_groups:
            #     print(param_group['lr'])


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, counter, loss_history, iteration_number):
    model.train()
    losses = []
    total_loss = 0
    conv_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target
        loss_outputs = loss_fn(*loss_inputs)
        loss_outputs = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

        # combine losses
        loss = loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            iteration_number += log_interval
            counter.append(iteration_number)
            loss_history.append(loss.item())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    conv_loss /= (batch_idx + 1)
    return total_loss, conv_loss, counter, loss_history, iteration_number


def test_epoch(val_loader, model, loss_fn, cuda):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        conv_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target
            loss_outputs = loss_fn(*loss_inputs)
            loss_outputs = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            # combine losses
            loss = loss_outputs
            val_loss += loss.item()
            conv_loss = 0

    return val_loss, conv_loss


def show_epoch_plot(epoch, plt_epoch, epoch_loss, plt_val_loss, path):
    fig = plt.figure()
    plt.plot(plt_epoch[1:], epoch_loss[1:])
    plt.plot(plt_epoch[1:], plt_val_loss[1:])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('epoch-{}'.format(epoch+1))
    plt.legend(['train', 'validation'])
    plt.show()
    fig.savefig(os.path.join(path, 'epoch-{}.tiff'.format(epoch+1)))
