# -*- coding= utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import copy
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(         # input shape (1, 128, 128)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=64,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (32, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),                      # activation
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ####################################################
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # ####################################################
            # nn.Conv2d(128, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
        )


    def forward(self, x):
        output = self.features(x)

        return output


#net1 = torchvision.models.vgg16_bn(pretrained=True, )
#print(net1)
#
# net2 = CNN()
# print(net2.state_dict())


def transfer_weights(model_from, model_to):
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys():
        if  k in wf:
            wt[k] = wf[k]

    model_to.load_state_dict(wt)
#
# transfer_weights(net1,net2)
#
# print(net2.state_dict())


def init_vgg16():
    net1 = torchvision.models.vgg16_bn(pretrained=True, )
    net2 = CNN()
    transfer_weights(net1, net2)
    return net2