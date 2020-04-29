import torch
import torch.nn as nn
from torchvision import models
from copy import deepcopy
import torch.nn.functional as F
from torch.autograd import Variable
import math


__all__ = ['stn_net']


def VGG16_initializator():
    layer_names = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3",
                   "conv4_1", "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3"]
    layers = list(models.vgg16_bn(pretrained=True).features.children())
    layers = [x for x in layers if isinstance(x, nn.Conv2d)]
    layer_dic = dict(zip(layer_names, layers))
    return layer_dic


def make_layers_from_names(names, model_dic, bn_dim, existing_layer=None):
    layers = []
    if existing_layer is not None:
        layers = [existing_layer, nn.BatchNorm2d(bn_dim, momentum=0.1), nn.ReLU(inplace=True)]
    for name in names:
        layers += [deepcopy(model_dic[name]), nn.BatchNorm2d(bn_dim, momentum=0.1), nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)


N_PARAMS = {'affine': 6,
            'rotation': 1}


# Spatial transformer network forward function
def stn(x, theta, mode='rotation', reduce_ratio=150/224):
    rr = reduce_ratio
    angle = None
    if mode == 'affine':
        theta1 = theta.view(-1, 2, 3)
    else:
        theta1 = Variable(torch.zeros([x.size(0), 2, 3], dtype=torch.float32, device=x.get_device()),
                          requires_grad=True)
        theta1 = theta1 + 0
        theta1[:, 0, 0] = 1.0
        theta1[:, 1, 1] = 1.0
        if mode == 'rotation':
            angle = theta[:, 0]
            theta1[:, 0, 0] = torch.cos(angle) * rr
            theta1[:, 0, 1] = -torch.sin(angle) * rr
            theta1[:, 1, 0] = torch.sin(angle) * rr
            theta1[:, 1, 1] = torch.cos(angle) * rr

    target_size = [x.size(0), x.size(1), 150, 150]
    grid = F.affine_grid(theta1, target_size)
    x = F.grid_sample(x, grid)
    return x, angle


class EmbeddingNet(nn.Module):
    def __init__(self, stn_mode='rotation'):
        super(EmbeddingNet, self).__init__()

        model_dic = VGG16_initializator()

        self.CBR1_ENC = make_layers_from_names(["conv1_1", "conv1_2"], model_dic, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR2_ENC = make_layers_from_names(["conv2_1", "conv2_2"], model_dic, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR3_ENC = make_layers_from_names(["conv3_1", "conv3_2", "conv3_3"], model_dic, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR4_ENC = make_layers_from_names(["conv4_1", "conv4_2", "conv4_3"], model_dic, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR5_ENC = make_layers_from_names(["conv5_1", "conv5_2", "conv5_3"], model_dic, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stn_mode = stn_mode
        self.stn_n_params = N_PARAMS[stn_mode]

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.stn_n_params),
            nn.Tanh()
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[3].weight.data.fill_(0)
        self.fc_loc[3].weight.data.zero_()
        if self.stn_mode == 'affine':
            self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif self.stn_mode == 'rotation':
            self.fc_loc[3].bias.data.copy_(torch.tensor([0], dtype=torch.float))

    def theta(self, x):
        xs = self.pool1(self.CBR1_ENC(x))
        xs = self.pool2(self.CBR2_ENC(xs))
        xs = self.pool3(self.CBR3_ENC(xs))
        xs = self.pool4(self.CBR4_ENC(xs))
        xs = self.pool5(self.CBR5_ENC(xs))
        xs = xs.view(-1, 512 * 7 * 7)
        theta = self.fc_loc(xs)  # for rotation: only 1 param, it is the angle
        theta = theta * math.pi * 1.5
        return theta

    def forward(self, x):
        # transform the input
        theta = self.theta(x)

        x, angle = stn(x, theta, mode=self.stn_mode, reduce_ratio=28/224)
        return x, angle


class RotStnNet(nn.Module):
    def __init__(self, embedding_net):
        super(RotStnNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        feat0, theta0 = self.embedding_net(x1)
        feat1, theta1 = self.embedding_net(x2)
        return (feat0, theta0), (feat1, theta1)


def stn_net():
    embedding_net = EmbeddingNet()
    model = RotStnNet(embedding_net)
    return model


