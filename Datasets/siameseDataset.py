import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import random
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import math


class SiameseDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = self.dataset.transform

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = self.dataset.imgs[index]
        # same class
        if target == 1:
            while True:
                # keep looping till the same class image is found
                img2, label2 = random.choice(self.dataset.imgs)
                if label2 == label1 and img1 != img2:
                    break
        # different class
        else:
            while True:
                # keep looping till a different class image is found
                img2, label2 = random.choice(self.dataset.imgs)
                if label2 != label1:
                    break

        img1 = Image.open(img1)
        img2 = Image.open(img2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), target

    def __len__(self):
        return len(self.dataset.imgs)


class SiameseAugDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = self.dataset.transform
        self.angle_choice = [0]

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = self.dataset.imgs[index]
        # same class
        if target == 1:
            while True:
                # keep looping till the same class image is found
                img2, label2 = random.choice(self.dataset.imgs)
                if label2 == label1 and img1 != img2:
                    break
        # different class
        else:
            while True:
                # keep looping till a different class image is found
                img2, label2 = random.choice(self.dataset.imgs)
                if label2 != label1:
                    break

        img1 = Image.open(img1)
        img2 = Image.open(img2)

        # add rotation during training siamese net
        label1 = random.choice(self.angle_choice)
        img1 = TF.rotate(img1, label1)
        label2 = random.choice([z for z in self.angle_choice if z != label2])
        img2 = TF.rotate(img2, label2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), target

    def __len__(self):
        return len(self.dataset.imgs)

pi = math.pi
IMG_EXTENSIONS = '.tif'


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, str(target))
                    images.append(item)
    return images


class DataAugSiaDataset(Dataset):
    # input images before transforms is 370*370
    def __init__(self, root, extensions=IMG_EXTENSIONS, is_valid_file=None):
        self.root = root
        self.transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.angle_choice = [0]

    def _find_classes(self, dir):
            """
            Finds the class folders in a dataset.
            Args:
                dir (string): Root directory path.
            Returns:
                tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
            Ensures:
                No class is a subdirectory of another.
            """
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return classes, class_to_idx

    def __getitem__(self, index):
        img0, label0 = self.samples[index]
        target = np.random.randint(0, 2)

        # while True:
        #     # keep looping till the same class image is found
        #     img1, label1 = random.choice(self.samples)
        #     if label1 == label0 and img1 != img0:
        #         break

        if target == 1:
            while True:
                # keep looping till the same class image is found
                img1, label1 = random.choice(self.samples)
                if label1 == label0 and img1 != img0:
                    break
        # different class
        else:
            while True:
                # keep looping till a different class image is found
                img1, label1 = random.choice(self.samples)
                if label1 != label0:
                    break

        img0 = Image.open(img0)
        img1 = Image.open(img1)
        # img0.show()
        # img1.show()

        # angle_label0 = random.choice(self.angle_choice)
        # img0 = TF.rotate(img0, angle_label0)
        # angle_label1 = random.choice([z for z in self.angle_choice if z != angle_label0])
        # img1 = TF.rotate(img1, angle_label1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # angle_label0 = angle_label0*pi/180.
        # angle_label1 = angle_label1*pi/180.

        return (img0, img1), target

    def __len__(self):
        return len(self.samples)
