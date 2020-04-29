import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


__all__ = ['siamese_net']


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
                    *list(models.vgg16_bn(pretrained=True).features)
                )
        self.fc = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(4096, 4096))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2


def siamese_net():
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)
    return model

