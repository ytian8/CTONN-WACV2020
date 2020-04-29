import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torchvision.transforms as transforms

import torch
import torchvision.transforms.functional as TF
import random
import math

pi = math.pi


class Stn4ClassesDataset(Dataset):
    # input images before transforms is 256*256
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = self.dataset.transform
        self.angle_choice = [0, 90, 180, 270]

    def __getitem__(self, index):
        img, _ = self.dataset.imgs[index]

        img0 = Image.open(img)
        img1 = Image.open(img)

        label0 = random.choice(self.angle_choice)
        img0 = TF.rotate(img0, label0)
        label1 = random.choice([z for z in self.angle_choice if z != label0])
        img1 = TF.rotate(img1, label1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        label0 = label0*pi/180.
        label1 = label1*pi/180.

        return (img0, label0), (img1, label1)

    def __len__(self):
        return len(self.dataset.imgs)


class StnNClassesDataset(Dataset):
    # input images is 370*370
    def __init__(self, dataset, Nclasses):
        self.dataset = dataset
        self.rotation_classes = Nclasses
        self.transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if self.rotation_classes == 8:
            self.angle_choice = [0, 45, 90, 135, 180, 225, 270, 315]
        if self.rotation_classes == 36:
            self.angle_choice = [z for z in range(0, 360, 10)]

    def __getitem__(self, index):
        img, _ = self.dataset.imgs[index]

        label0 = random.choice(self.angle_choice)
        img0 = TF.rotate(Image.open(img), label0)
        label1 = random.choice([z for z in self.angle_choice if z != label0])
        img1 = TF.rotate(Image.open(img), label1)

        img0 = self.transform(img0)
        img1 = self.transform(img1)

        label0 = label0*pi/180.
        label1 = label1*pi/180.

        return (img0, label0), (img1, label1)

    def __len__(self):
        return len(self.dataset.imgs)


class Stn4ClassesDatasetDiffYear(Dataset):
    # input images before transforms is 256*256
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = self.dataset.transform
        self.angle_choice = [0, 90, 180, 270]

    def __getitem__(self, index):
        img0, label0 = self.dataset.imgs[index]

        while True:
            # keep looping till the same class image is found
            img1, label1 = random.choice(self.dataset.imgs)
            if label1 == label0 and img1 != img0:
                break

        img0 = Image.open(img0)
        img1 = Image.open(img1)
        # img0.show()
        # img1.show()

        angle_label0 = random.choice(self.angle_choice)
        img0 = TF.rotate(img0, angle_label0)
        angle_label1 = random.choice([z for z in self.angle_choice if z != angle_label0])
        img1 = TF.rotate(img1, angle_label1)
        # img0.show()
        # img1.show()

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        angle_label0 = angle_label0*pi/180.
        angle_label1 = angle_label1*pi/180.

        return (img0, angle_label0), (img1, angle_label1)

    def __len__(self):
        return len(self.dataset.imgs)


class StnNClassesDatasetDiffYear(Dataset):
    # input images before transforms is 256*256
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.angle_choice = [z for z in range(0, 360, 10)]

    def __getitem__(self, index):
        img0, label0 = self.dataset.imgs[index]

        while True:
            # keep looping till the same class image is found
            img1, label1 = random.choice(self.dataset.imgs)
            if label1 == label0 and img1 != img0:
                break

        img0 = Image.open(img0)
        img1 = Image.open(img1)
        # img0.show()
        # img1.show()

        angle_label0 = random.choice(self.angle_choice)
        img0 = TF.rotate(img0, angle_label0)
        angle_label1 = random.choice([z for z in self.angle_choice if z != angle_label0])
        img1 = TF.rotate(img1, angle_label1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        angle_label0 = angle_label0*pi/180.
        angle_label1 = angle_label1*pi/180.

        return (img0, angle_label0), (img1, angle_label1)

    def __len__(self):
        return len(self.dataset.imgs)