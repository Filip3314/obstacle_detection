from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from numpy.typing import NDArray
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import ResNet as rn
import VGGNet as vgn
import New_FC as nfc
import dataset


# I am going to write something so that it can train all three of the models
# At the same time. I think this is the best way to make sure that they are
# All equivalent
def train_models():
    # Load in the datasets
    # TODO add the appropriate file paths once they are available
    # TODO make sure that the training and validation data re appropriately split
    train_data = dataset.ClassificationDataset()
    #val_data = dataset.ClassificationDataset()
    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    #val_dl = DataLoader(val_data, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    # Now, I am going to create a silly little array of models so I can train
    # All of them at once
    # Adding a functionality to recall what type of model it is so it can do
    # the image slicing in the forward pass and I can just for loop all of them

    # Resnets first
    resnet_RGB_model = rn.ResNet(img_type='RGB').to('cpu')
    resnet_RGB_optim = torch.optim.Adam(resnet_RGB_model.parameters())

    resnet_depth_model = rn.ResNet(img_type='depth').to('cpu')
    resnet_depth_optim = torch.optim.Adam(resnet_depth_model.parameters())

    resnet_RGBD_model = rn.ResNet(img_type='RGBD').to('cpu')
    resnet_RGBD_optim = torch.optim.Adam(resnet_RGBD_model.parameters())

    # VGGnets
    vggnet_RGB_model = vgn.VGGNet(img_type='RGB').to('cpu')
    vggnet_RGB_optim = torch.optim.Adam(vggnet_RGB_model.parameters())

    vggnet_depth_model = vgn.VGGNet(img_type='depth').to('cpu')
    vggnet_depth_optim = torch.optim.Adam(vggnet_depth_model.parameters())

    vggnet_RGBD_model = vgn.VGGNet(img_type='RGBD').to('cpu')
    vggnet_RGBD_optim = torch.optim.Adam(vggnet_RGBD_model.parameters())

    #FCs
    fc_RGB_model = nfc.New_FC(img_type='RGB').to('cpu')
    fc_RGB_optim = torch.optim.Adam(fc_RGB_model.parameters())

    fc_depth_model = nfc.New_FC(img_type='depth').to('cpu')
    fc_depth_optim = torch.optim.Adam(fc_depth_model.parameters())

    fc_RGBD_model = nfc.New_FC(img_type='RGBD').to('cpu')
    fc_RGBD_optim = torch.optim.Adam(fc_RGBD_model.parameters())

    # Creates a list of these so I can  iterate through it on each training step
    models = [resnet_RGB_model, resnet_depth_model, resnet_RGBD_model,
              vggnet_RGB_model, vggnet_depth_model, vggnet_RGBD_model,
              fc_RGB_model, fc_depth_model, fc_RGBD_model]
    optims = [resnet_RGB_optim, resnet_depth_optim, resnet_RGBD_optim,
              vggnet_RGB_optim, vggnet_depth_optim, vggnet_RGBD_optim,
              fc_RGB_optim, fc_depth_optim, fc_RGBD_optim]

    # Initialize the training and validation
    # Currently running 20 steps, perhaps will do more later
    pbar = tqdm(range(1, 20 + 1))
    record = dict(train_loss=[], val_loss=[])
    for epoch_num in pbar:
        # First set all of the models to the training mode
        for model in models:
            model.train()

        # Now, we will iterate through the batch provided by the dataloader
        for batch in train_dl:
            # Not entirely sure what this does but it seems important idk
            img, label = map(lambda x: x.to('cpu'), batch)

            # This will train each of the models on the same batch
            for i in range(len(models)):
                logits = models[i].forward(img)
                loss = criterion(logits, label)

                optims[i].zero_grad()
                loss.backward()  # Not sure what this does
                optims[i].step()

        # Now, we will do the validation step
        # Also, will save the weights here
        for model in models:
            model.save()
            model.eval()

        #for batch in val_dl:
        #    img, label = map(lambda x: x.to('cpu'), batch)

