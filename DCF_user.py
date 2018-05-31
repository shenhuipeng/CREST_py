# -*- coding= utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import copy
import torch.optim as optim
from torch.autograd import Variable
import cv2
from time import time
import matplotlib.pyplot as plt


num_channels = 64

class DCF_conv(nn.Module):
    def __init__(self,size,num_channels):
        super(DCF_conv, self).__init__()

        padding_uesr = (size-1)//2
        self.bn = nn.BatchNorm2d(num_channels)
        self.dcf = nn.Sequential(         # input shape (1, 128, 128)
            #nn.BatchNorm2d(num_channels),
            nn.Conv2d(
                in_channels=num_channels,              # input height
                out_channels=1,            # n_filters
                kernel_size=(size[0],size[1]),              # filter size
                stride=1,                   # filter movement/step
                padding=(padding_uesr[0],padding_uesr[1]),                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (32, 128, 128)
        )
        self.res1 = nn.Sequential(
            # nn.BatchNorm2d(num_channels),
            nn.Conv2d(num_channels,num_channels,1,1,0),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(num_channels, 1, 1, 1, 0)
        )
        self.res2 = nn.Sequential(
            #nn.BatchNorm2d(num_channels),
            nn.Conv2d(num_channels, 1, 1, 1, 0)
        )
        #self.softmax2d =  nn.Softmax2d()



    def forward(self, x):

        x = self.bn(x)
        output = self.dcf(x)+self.res1(x)+self.res2(x)

        #output = self.softmax2d(output)*10



        return output



def test():
    feature = np.load("./feature.npy")
    (fh, fw, fc)= feature.shape

    print (feature.shape)
    label = np.load("./label.npy")*10
    print(label.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(label)
    plt.show()
    size = np.array([fh, fw])
    model = DCF_conv(size,fc)
    print (model)
    feature = feature[np.newaxis,:,:,:]
    label = label[np.newaxis,:,:,np.newaxis]
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-6)
    loss_fn = nn.MSELoss(reduce=True, size_average=True)

    model.train()
    for batch_idx in range(300):
        data, target = feature, label
        # print(data.shape)
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        # convert BHWC to BCHW
        data = data.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
        #print(data)
        data, target = data.float().cuda(), target.float().cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        loss = loss_fn(output, target) #+ 1e-6*output.norm(2)
        loss.backward()
        optimizer.step()
        if batch_idx%100==0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(batch_idx, loss.item()))
    output = output.cpu().detach().numpy()
    output = output[0,0,:,:]
    #print(output)
    # cv2.imshow("",output)
    # cv2.waitKey(0)
    #fig = plt.figure()
    # 第一个子图,按照默认配置
    #ax = fig.add_subplot(111)
    #ax.imshow(output)
    #plt.show()
    #fig = plt.figure()
    test_input = np.roll(feature,100,axis=1)
    test_input = np.roll(test_input, 100, axis=2)
    #ax = fig.add_subplot(111)
    #ax.imshow(test_input[0,:,:,:])
    #plt.show()
    test_label = np.roll(label,100,axis=1)
    test_label = np.roll(test_label, 100, axis=2)
    data, target = test_input, test_label

    data, target = torch.from_numpy(data), torch.from_numpy(target)
    # convert BHWC to BCHW
    data = data.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)
    # print(data)
    data, target = data.float().cuda(), target.float().cuda()

    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    output = output.cpu().detach().numpy()

    fig = plt.figure()
    # 第一个子图,按照默认配置
    ax = fig.add_subplot(121)
    ax.imshow(test_label[0,:,:,0])
    ay = fig.add_subplot(122)
    ay.imshow(output[0,0, :, :])
    plt.show()

    

#test()


class DCF_layer():
    def __init__(self,feature_map,label):
        (fh, fw, fc) = feature_map.shape
        print("Start init DCF_layer...")
        t1=time()
        print("feature map shape:", feature_map.shape)
        print("label shape:",label.shape)
        size = np.array([fh, fw])
        feature_map = feature_map[np.newaxis, :, :, :]
        label = label[np.newaxis, :, :, np.newaxis]

        self.dcf_model = DCF_conv(size, fc)
        print(self.dcf_model)
        self.model = self.dcf_model.cuda()
        self.optimizer = optim.Adam(self.dcf_model.parameters(), lr=1e-4, weight_decay=1e-6)
        self.loss_fn = nn.MSELoss(reduce=True, size_average=True)

        self.dcf_model.train()

        data, target = feature_map, label
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        # convert BHWC to BCHW
        data = data.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

        for batch_idx in range(300):

            data, target = data.float().cuda(), target.float().cuda()

            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.dcf_model(data)

            loss = self.loss_fn(output, target)  # + 1e-6*output.norm(2)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {}\tLoss: {:.6f}'.format(batch_idx, loss.item()))
        output = output.cpu().detach().numpy()
        t2=time()
        print("DCF_layer init end")
        print("init time cost:",(t2-t1)/1000 , "s")
        