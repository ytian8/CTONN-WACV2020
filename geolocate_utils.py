import os
import models
import torch
import glob
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16_bn
import torchvision.datasets as datasets
from featExtr_eval import evaluation


def main_geolocate(args):
    if args.get_feats_mode == 'sift':
        accuracy = evaluation.feats_eval_sift(args)
    else:
        # 1. load pre-trained feature extractor model
        if args.get_feats_mode == 'split':
            if args.runtime_arch == 'vgg':
                model = nn.Sequential(*list(vgg16_bn(pretrained=True).features[:34]))
            else:
                model = load_model(args)
        if args.get_feats_mode == 'global':
            if args.runtime_arch == 'vlad_siamese_net':
                model = models.__dict__[args.runtime_arch]()
                model.load_state_dict(torch.load(args.runtime_arch_trained_path))
                model = model.embedding_net
            if args.runtime_arch == 'vgg_fc7':
                model = vgg16_bn(pretrained=True)
                model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
            if args.runtime_arch == 'vgg_fc6':
                model = vgg16_bn(pretrained=True)
                model.classifier = nn.Sequential(*[model.classifier[i] for i in range(1)])

        # 2. Data loading code
        transform = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        target_dataset = datasets.ImageFolder(root=args.target_data, transform=transform)
        query_dataset = datasets.ImageFolder(root=args.query_data, transform=transform)

        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=1, pin_memory=True)
        query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=1, shuffle=False,
                                                   num_workers=1, pin_memory=True)

        # 3. geo-localization
        accuracy = evaluation.featsExtr_eval(args, model, transform, query_loader, target_loader)

        # overlap_table = '/home/yuxin/PycharmProjects/wacv2020_2nd/featExtr_eval/la_cross_overlap_table.csv'
        # evaluation.get_overlap_hisgram(args, model, transform, query_loader, target_loader, overlap_table)

        print('stn:{}\tstn_trained_path:{}\tsiamese:{}\ttrained_path:{}\t'.format(
            args.stn_net, args.stn_trained_path, args.siamese_net, args.siamese_trained_path))
    # return accuracy


# args.runtime_arch, args.stn_trained_path, args.siamese_trained_path
def load_model(args):
    runtime_model = models.__dict__[args.runtime_arch]()

    runtime_model_dict = runtime_model.state_dict()
    # print(runtime_model.CBR1_ENC[0].weight)

    # print("======================================================")
    # print(runtime_model.siameseFeatExtr[0].weight)

    # load stn
    stn_net = models.__dict__[args.stn_net]()
    stn_net.load_state_dict(torch.load(args.stn_trained_path))
    stn_pretrained_dict = stn_net.embedding_net.state_dict()

    # 1). filter out unnecessary keys for stn
    stn_pretrained_dict = {k: v for k, v in stn_pretrained_dict.items() if k in runtime_model_dict}
    # 2). overwrite entries in the existing state dict
    runtime_model_dict.update(stn_pretrained_dict)
    # 3). load the new state dict
    runtime_model.load_state_dict(runtime_model_dict)
    # print(runtime_model.CBR1_ENC[0].weight)

    # load siamese feature extractor
    if args.siamese_net == 'siamese_net':
        siamese_net = models.__dict__[args.siamese_net]()
        siamese_net.load_state_dict(torch.load(args.siamese_trained_path))
        siamese_modules = list(siamese_net.embedding_net.convnet[:34])

    elif args.siamese_net == 'siamese_net_2':
        siamese_net = models.__dict__[args.siamese_net]()
        siamese_net.load_state_dict(torch.load(args.siamese_trained_path))
        siamese_modules = list(siamese_net.embedding_net.conv4)
    else:
        raise ValueError('Wrong siamese feature extractor arch')

    runtime_model.siameseFeatExtr = nn.Sequential(*list(siamese_modules))
    # print("======================================================")
    # print(runtime_model.siameseFeatExtr[0].weight)

    return runtime_model

