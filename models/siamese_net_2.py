import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


__all__ = ['siamese_net_2']


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.conv4 = nn.Sequential(
                    *list(models.vgg16_bn(pretrained=True).features[:34])
        )
        self.conv5 = nn.Sequential(
            *list(models.vgg16_bn(pretrained=True).features[34:])
        )
        self.fc = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(4096, 4096))

    def forward(self, x):
        conv4_out = self.conv4(x)
        conv4_avgpool = torch.mean(conv4_out, 1)
        conv5_out = self.conv5(conv4_out)
        output = conv5_out.view(conv5_out.size()[0], -1)
        output = self.fc(output)
        return conv4_avgpool.view(conv4_out.size()[0], -1), output


class SiameseNetTwoLoss(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNetTwoLoss, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        conv4_out1, output1 = self.embedding_net(x1)
        conv4_out2, output2 = self.embedding_net(x2)
        return (conv4_out1, conv4_out2), (output1, output2)


def siamese_net_2():
    embedding_net = EmbeddingNet()
    model = SiameseNetTwoLoss(embedding_net)
    return model

