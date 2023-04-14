from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from numpy.typing import NDArray
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vgg


class New_FC(nn.Module):
    def __init__(self, img_type="RGB"):
        super().__init__()

        # I am going to do the slicing of the image in the forward pass, so I
        # will need to save the image type
        self.img_type = img_type

        # Start by flattening the input image
        image_flatten = nn.Flatten()

        # Num features will be determined by the size of the input image
        if img_type == "RGB":
            num_features = 256 * 256 * 3
        if img_type == "depth":
            num_features = 256 * 256
        if img_type == "RGBD":
            num_features = 256 * 256 * 4

        # Define fully connected layers
        relu = nn.ReLU()
        first_layer = nn.Linear(in_features=num_features, out_features=100)
        second_layer = nn.Linear(in_features=100, out_features=100)
        third_layer = nn.Linear(in_features=100, out_features=100)
        fourth_layer = nn.Linear(in_features=100, out_features=100)
        fifth_layer = nn.Linear(in_features=100, out_features=100)
        output_classifier = nn.Linear(in_features=100, out_features=100)

        self.model = nn.Sequential(image_flatten, first_layer, relu, second_layer, relu,
                                   third_layer, relu, fourth_layer, relu, fifth_layer, relu, output_classifier)
        self.save_path = "New_FC_" + img_type + ".pt"

    def forward(self, img: Tensor) -> Tensor:
        return self.model(img)

    def save(self) -> None:
        '''Save model weights to *.pt file'''
        torch.save(self.state_dict(), self.save_path)

    def load(self) -> None:
        '''Load model weights from *.pt file'''
        self.load_state_dict(torch.load(self.save_path))
