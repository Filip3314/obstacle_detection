import os
import torch

from torch.utils.data import Dataset
from PIL import Image


class ClassificationDataset(Dataset):

    def __init__(self, rgb_dir, depth_dir, labels, transform = None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        # Must have the same number of depth and rgb images to have a valid dataset
        assert len(rgb_dir) == len(depth_dir)
        self.rgb_filenames = os.listdir(rgb_dir)
        self.depth_filenames = os.listdir(depth_dir)
        self.labels = labels

    def __len__(self):
        return len(self.rgb_filenames)

    def __getitem__(self, item: int):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_filenames[item])
        depth_path = os.path.join(self.depth_dir, self.depth_filenames[item])
        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path).convert("L") # grayscale

        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)

        merged_image = torch.cat((rgb_image, depth_image.unsqueeze(0)), dim=0)

        return merged_image


class SegmentationDataset(ClassificationDataset):

    def __init__(self, rgb_dir, depth_dir, transform = None):
        super().__init__(rgb_dir,depth_dir,transform)

    def __len__(self):
        return len(self.rgb_filenames)

    def __getitem__(self, item: int):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_filenames[item])
        depth_path = os.path.join(self.depth_dir, self.depth_filenames[item])
        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path).convert("L") # grayscale

        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)

        merged_image = torch.cat((rgb_image, depth_image.unsqueeze(0)), dim=0)

        return merged_image
