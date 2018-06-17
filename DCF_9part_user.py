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
from sklearn.decomposition import PCA


class DCF_part_conv(nn.Module):
    def __init__(self, size, part_num,num_channels):
        super(DCF_part_conv, self).__init__()

        padding_uesr = (size - 1) // 2
        # self.bn = nn.BatchNorm2d(num_channels)
        self.dcf = nn.Sequential(  # input shape (1, 128, 128)
            # nn.BatchNorm2d(num_channels),
            nn.Conv2d(
                in_channels=num_channels,  # input height
                out_channels=part_num,  # n_filters
                kernel_size=(size[0], size[1]),  # filter size
                stride=1,  # filter movement/step
                padding=(padding_uesr[0], padding_uesr[1]),
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),

            # output shape (32, 128, 128)
        )
        self.res1 = nn.Sequential(
            # nn.BatchNorm2d(num_channels),
            nn.Conv2d(num_channels, num_channels // 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num_channels // 2, num_channels // 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num_channels // 4,part_num , 3, 1, 1),

            nn.PReLU(),
        )
        # self.res2 = nn.Sequential(
        #     # nn.BatchNorm2d(num_channels),
        #     nn.Conv2d(num_channels, 1, 1, 1, 0),
        #
        #     nn.PReLU(),
        # )
        # self.softmax2d =  nn.Softmax2d()

    def forward(self, x):
        # x = self.bn(x)
        output = self.dcf(x) + self.res1(x) #+ self.res2(x)

        return output


class DCF_part_layer():
    def __init__(self, featureNet, image_gen, label_gen, target_h_w, train_num, channel_num, cos_window=None):
        t1 = time()
        self.channel_num = channel_num
        self.cos_window = cos_window
        self.image_gen = image_gen
        self.label_gen = label_gen
        image_batch = next(self.image_gen)
        label_batch = next(self.label_gen)
        num, fh, fw, fc = image_batch.shape

        featureNet_input = torch.from_numpy(image_batch)
        featureNet_input = featureNet_input.permute(0, 3, 1, 2)
        featureNet_input = featureNet_input.float().cuda()
        label_batch = (torch.from_numpy(label_batch))
        label_batch = label_batch.permute(0, 3, 1, 2)
        label_batch = label_batch.float().cuda()
        featureNet_output = featureNet(featureNet_input)

        # featureNet_output = self.pca(featureNet_output,channel_num=self.channel_num)
        featureNet_output = featureNet_output[:, ::(256 // self.channel_num), :, :]
        if self.cos_window != None:
            window_feature = np.zeros(featureNet_output.shape)
            for i in range(self.channel_num):
                for j in range(num):
                    window_feature[j, :, :, i] = featureNet_output[j, :, :, i] * self.cos_window
            featureNet_output = window_feature

        print("##########")
        print("Start init DCF_layer...")

        print("feature map shape:", featureNet_output.shape)
        print("label shape:", label_batch.shape)

        th = np.ceil(target_h_w[0] / 2)
        tw = np.ceil(target_h_w[1] / 2)
        th = 2 * th + 1
        tw = 2 * tw + 1
        # size = np.array([max(th, tw),max(th, tw)])
        size = np.array([th, tw])

        self.dcf_model = DCF_conv(size, channel_num)
        print(self.dcf_model)
        self.model = self.dcf_model.cuda()
        self.optimizer = optim.Adam(self.dcf_model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss(reduce=True, size_average=True)

        self.dcf_model.train()

        data, target = Variable(featureNet_output), Variable(label_batch)
        self.optimizer.zero_grad()
        output = self.dcf_model(data)
        loss = self.loss_fn(output, target)  # + 1e-6*output.norm(2)
        loss.backward()
        self.optimizer.step()

        for i in range(1, train_num):
            image_batch = next(self.image_gen)
            label_batch = next(self.label_gen)
            featureNet_input = torch.from_numpy(image_batch)
            featureNet_input = featureNet_input.permute(0, 3, 1, 2)
            featureNet_input = featureNet_input.float().cuda()
            label_batch = (torch.from_numpy(label_batch))
            label_batch = label_batch.permute(0, 3, 1, 2)
            label_batch = label_batch.float().cuda()
            featureNet_output = featureNet(featureNet_input)
            # featureNet_output = self.pca(featureNet_output, channel_num=self.channel_num)
            featureNet_output = featureNet_output[:, ::(256 // self.channel_num), :, :]
            data, target = Variable(featureNet_output), Variable(label_batch)
            self.optimizer.zero_grad()
            output = self.dcf_model(data)
            self.optimizer.zero_grad()
            output = self.dcf_model(data)
            loss = self.loss_fn(output, target)  # + 1e-6*output.norm(2)
            loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                print('Train num: {}\tLoss: {:.6f}'.format(i, loss.item()))
        sample = output.cpu().detach().numpy()
        sample = sample[0, 0, :, :]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(sample)
        # plt.show()
        # plt.savefig(str(time())+".jpg")
        plt.pause(2)  # 显示秒数
        plt.close()

        t2 = time()
        print("DCF_layer init end")
        print("init time cost:", (t2 - t1), "s")
        print("##########")

    def search(self, feature_map):
        self.dcf_model.eval()
        t1 = time()
        #         feature_map = feature_map[np.newaxis, :, :, :]
        #         data = torch.from_numpy(feature_map)
        #         data = data.permute(0, 3, 1, 2)
        #         data = data.float().cuda()
        feature_map = feature_map[:, ::(256 // self.channel_num), :, :]
        output = self.dcf_model(feature_map)
        output = output.permute(0, 2, 3, 1)
        output = output.cpu().detach().numpy()
        # output = output[0, 0, :, :]
        t2 = time()
        return output

    def pca(self, feature_map, channel_num):
        feature_map = feature_map.permute(0, 2, 3, 1)
        feature_map = feature_map.cpu().detach().numpy()
        batch_num, hf, wf, cf = feature_map.shape
        matrix = np.reshape(feature_map, (batch_num, hf * wf, cf))
        coeff = np.zeros((batch_num, hf * wf, channel_num))
        for i in range(batch_num):
            pca = PCA(n_components=channel_num)
            pca.fit(matrix[i, :, :])
            coeff[i, :, :] = pca.transform(matrix[i, :, :])
        # print("after pca", np.shape(coeff))
        featurePCA = np.reshape(coeff, (batch_num, hf, wf, -1))
        # featurePCA = np.zeros((batch_num, hf ,wf,channel_num))
        # featurePCA[:,:,:,:]= feature_map[:,:,:,0:channel_num]
        featurePCA = torch.from_numpy(featurePCA)
        featurePCA = featurePCA.permute(0, 3, 1, 2)
        featurePCA = featurePCA.float().cuda()

        return featurePCA

    def update(self, feature_batch, label_betch):
        self.dcf_model.train()
        t1 = time()
        # featureNet_input = torch.from_numpy(feature_batch)
        # featureNet_input = featureNet_input.permute(0, 3, 1, 2)
        # featureNet_input = featureNet_input.float().cuda()
        label_batch = (torch.from_numpy(label_betch))
        label_batch = label_batch.permute(0, 3, 1, 2)
        label_batch = label_batch.float().cuda()
        # featureNet_output = featureNet(featureNet_input)

        # featureNet_output = self.pca(featureNet_output,channel_num=self.channel_num)
        featureNet_output = feature_batch[:, ::(256 // self.channel_num), :, :]
        data, target = Variable(featureNet_output), Variable(label_batch)
        for i in range(5):
            self.optimizer.zero_grad()
            output = self.dcf_model(data)
            self.optimizer.zero_grad()
            output = self.dcf_model(data)
            loss = self.loss_fn(output, target)  # + 1e-6*output.norm(2)
            loss.backward()
            self.optimizer.step()

        t2 = time()

        print("update_time", t2 - t1)

    def re_init(self, featureNet):
        self.dcf_model.train()
        image_batch = next(self.image_gen)
        label_batch = next(self.label_gen)
        featureNet_input = torch.from_numpy(image_batch)
        featureNet_input = featureNet_input.permute(0, 3, 1, 2)
        featureNet_input = featureNet_input.float().cuda()
        label_batch = (torch.from_numpy(label_batch))
        label_batch = label_batch.permute(0, 3, 1, 2)
        label_batch = label_batch.float().cuda()
        featureNet_output = featureNet(featureNet_input)
        # featureNet_output = self.pca(featureNet_output, channel_num=self.channel_num)
        featureNet_output = featureNet_output[:, ::(256 // self.channel_num), :, :]
        data, target = Variable(featureNet_output), Variable(label_batch)
        self.optimizer.zero_grad()
        output = self.dcf_model(data)
        self.optimizer.zero_grad()
        output = self.dcf_model(data)
        loss = self.loss_fn(output, target)  # + 1e-6*output.norm(2)
        loss.backward()
        self.optimizer.step()