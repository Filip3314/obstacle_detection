from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from numpy.typing import NDArray
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet
from torchvision.models import resnet34, ResNet34_Weights
from torch.utils.data import Dataset


class ResNet(nn.Module):
    def __init__(self, img_type="RGB"):
        super().__init__()

        # I am going to do the slicing of the image in the forward pass, so I
        # will need to save the image type
        self.img_type = img_type

        weights = ResNet34_Weights
        self.pre_trained_model = resnet34(weights=weights)
        for param in self.pre_trained_model.parameters():
            param.requires_grad = False

        # This will alter the output class so we have a trainable output with our classes
        self.pre_trained_model.fc = nn.Linear(in_features=512, out_features=24)

        # Now we will handle the different image types - will only alter the first layer, the rest will be the same
        if img_type == "RGB":
            first_layer = nn.Conv2d(3, 64, kernel_size=(6, 6), stride=(2, 2), padding=(1, 1), bias=False)
        if img_type == "depth":
            first_layer = nn.Conv2d(1, 64, kernel_size=(6, 6), stride=(2, 2), padding=(1, 1), bias=False)
        if img_type == "RGBD":
            first_layer = nn.Conv2d(4, 64, kernel_size=(6, 6), stride=(2, 2), padding=(1, 1), bias=False)
            # first_layer = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.pre_trained_model.conv1 = first_layer
        # if img_type == "RGB":
        #    self.pre_trained_model.conv1.requires_grad = False

        self.save_path = "weights/ResNet_" + img_type + ".pt"

    def forward(self, img: Tensor) -> Tensor:
        # Going to slice up the image so I can take the channels that are relevant for this model
        # Assumes that the color channels are in dim 1 since dim 0 should now be the batch. Check this
        # first if getting errors
        R_slice = torch.select(img, 1, 0)
        G_slice = torch.select(img, 1, 1)
        B_slice = torch.select(img, 1, 2)
        D_slice = torch.select(img, 1, 3)

        # RGBD images remain unchanged. This should handle re-assembling the tensor for the other
        # image types
        if self.img_type == "RGB":
            img = torch.cat((R_slice.unsqueeze(1), G_slice.unsqueeze(1), B_slice.unsqueeze(1)), dim=1)
        if self.img_type == "depth":
            img = D_slice.unsqueeze(1)

        # the img does need to be floats unfortunately and the only way to do this is to cast it to a new variable rip
        img_floats = img.float()

        return self.pre_trained_model(img_floats)

    def save(self) -> None:
        '''Save model weights to *.pt file'''
        torch.save(self.state_dict(), self.save_path)

    def load(self) -> None:
        '''Load model weights from *.pt file'''
        self.load_state_dict(torch.load(self.save_path))

