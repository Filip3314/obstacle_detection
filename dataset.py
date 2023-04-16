import ast
import os

import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class PybulletDataset(Dataset):
    def __init__(self, rgb_dir: str = 'data/rgb', depth_dir: str = 'data/depth', transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        # Must have the same number of depth and rgb images to have a valid dataset
        # assert len(rgb_dir) == len(depth_dir)
        self.rgb_filenames = sorted(os.listdir(rgb_dir))
        self.depth_filenames = sorted(os.listdir(depth_dir))

    def __len__(self):
        return len(self.rgb_filenames)

    def __getitem__(self, item: int):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_filenames[item])
        depth_path = os.path.join(self.depth_dir, self.depth_filenames[item])
        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path).convert("L")  # grayscale

        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)

        # convert the RGB and the depth images to tensors so they can be manipulated more easily
        im_to_tensor = transforms.PILToTensor()
        rgb_tensor = im_to_tensor(rgb_image)
        depth_tensor = im_to_tensor(depth_image)

        # Concatenate RGB and depth
        merged_image = torch.cat((rgb_tensor, depth_tensor), dim=0)

        return merged_image


class ClassificationDataset(PybulletDataset):
    """For use in training an image classification model on the PyBullet images. Each label is a matrix with length equal
    to the number of categories"""
    def __init__(self, rgb_dir: str = 'data/rgb', depth_dir: str = 'data/depth', labels_file: str = 'labels.txt',
                 transform=None):
        super().__init__(rgb_dir, depth_dir, transform)
        self.labels = []
        with open('labels.txt') as file:
            for line in file:
                self.labels.append(ast.literal_eval(line))

    def __getitem__(self, item: int):
        label = self.labels[item]

        # First I will take everything in the dict and put it into a list for convenience
        label_list = []
        for i in range(len(label)):
            label_list.append((label[i]))

        # Now, tensorize the list
        label_tensor = torch.Tensor(label_list)

        return super().__getitem__(item), label_tensor


class SegmentationDataset(PybulletDataset):
    """For use in training an Image Segmentation model on the PyBullet images. Labels matrices the same shape as images,
    with each pixel labelled with a category"""
    def __init__(self, rgb_dir: str = 'data/rgb', depth_dir: str = 'data/depth', label_dir: str = 'data/labels',
                 transform=None):
        super().__init__(rgb_dir, depth_dir, transform)
        self.label_dir = label_dir
        self.label_filenames = sorted(os.listdir(label_dir))

    def __getitem__(self, item: int):
        label_path = os.path.join(self.label_dir, self.label_filenames[item])
        label = np.loadtxt(label_path, dtype=int)
        return super().__getitem__(item), label

