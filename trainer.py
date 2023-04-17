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
def train_models(epochs=20, RGB=False, depth=True, RGBD=False):
    # Load in the datasets
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

    models = []
    optims = []

    # Creates a list of these so I can  iterate through it on each training step
    if RGB:
        models.append(resnet_RGB_model)
        models.append(vggnet_RGB_model)
        models.append(fc_RGB_model)

        optims.append(resnet_RGB_optim)
        optims.append(vggnet_RGB_optim)
        optims.append(fc_RGB_optim)

    if depth:
        models.append(resnet_depth_model)
        models.append(vggnet_depth_model)
        models.append(fc_depth_model)

        optims.append(resnet_depth_optim)
        optims.append(vggnet_depth_optim)
        optims.append(fc_depth_optim)

    if RGBD:
        models.append(resnet_RGBD_model)
        models.append(vggnet_RGBD_model)
        models.append(fc_RGBD_model)

        optims.append(resnet_RGBD_optim)
        optims.append(vggnet_RGBD_optim)
        optims.append(fc_RGBD_optim)

    # Initialize the training and validation
    # Currently running 20 steps, perhaps will do more later
    pbar = tqdm(range(1, epochs + 1))
    record = dict(train_loss=[], val_loss=[])
    for epoch_num in pbar:
        # First set all of the models to the training mode
        for model in models:
            model.train()
            model.rec_new_train_losses()

        # Now, we will iterate through the batch provided by the dataloader

        for batch in train_dl:
            # Applies the __getitem__ function to the batch to return the image and label
            # to('cpu') indicates that this tensor should be prepped for running on the CPU instead
            # of a GPU
            img, label = map(lambda x: x.to('cpu'), batch)

            # This will train each of the models on the same batch
            for i in range(len(models)):
                logits = models[i].forward(img)
                loss = criterion(logits, label)
                models[i].append_train_losses(loss.item())

                optims[i].zero_grad()
                loss.backward()
                optims[i].step()

        # Now, we will do the validation step
        # Also, will save the weights here
        for model in models:
            model.save()
            model.eval()
            model.rec_new_val_losses()

        #for batch in val_dl:
        #    img, label = map(lambda x: x.to('cpu'), batch)

        # Will need to do the validation here once we have a distinct
        # Validation data dataset

        # record all of the data in the set
        for model in models:
            model.record_train_val_loss()
            model.plot_losses()

