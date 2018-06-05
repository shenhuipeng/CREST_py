import numpy as np
import cv2
from add_noise import add_random_noise
import matplotlib.pyplot as plt

def get_init_patch_batch(patch,num,loc,with_random_noise=True, with_special_patch=True):
    fh, fw, fc = patch.shape
    batch_data = np.zeros((num, fh, fw, fc)).astype("uint8")
    batch_data[0, :, :, :] = patch
    for i in range(1, num):
        batch_data[i, :, :, :] =  add_random_noise(patch)
    if with_special_patch:
        new1_batch = np.random.randint(0,255,size=[num, fh, fw, fc]).astype("uint8")
        new2_batch = np.random.randint(1, size=[num, fh, fw, fc]).astype("uint8")
        for j in range(num):
            # print(batch_data[j,loc[0]:(loc[0]+loc[2]),loc[1]:(loc[1]+loc[3]),:])
            new1_batch[j,loc[0]:(loc[0]+loc[2]),loc[1]:(loc[1]+loc[3]),:] = batch_data[j,loc[0]:(loc[0]+loc[2]),loc[1]:(loc[1]+loc[3]),:]
            new2_batch[j,loc[0]:(loc[0]+loc[2]),loc[1]:(loc[1]+loc[3]),:] = batch_data[j,loc[0]:(loc[0]+loc[2]),loc[1]:(loc[1]+loc[3]),:]

        batch_data = np.append(batch_data, new1_batch, axis=0)
        batch_data = np.append(batch_data, new2_batch, axis=0)
        #batch_data = np.append(new1_batch, new2_batch, axis=0)
    for i  in range(batch_data.shape[0]):
        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        ax.imshow(batch_data[i])
        plt.show()

    return batch_data


def get_init_batch(feature_map, label):
    num, fh, fw, fc = feature_map.shape
    train_data = np.zeros((num, fh, fw, fc))
    train_label = np.zeros((num, fh, fw))
    train_data[0, :, :, :] = feature_map[0, :, :, :]
    train_label[0, :, :] = label
    for i in range(1, num):
        h_shift = np.random.randint(1, fh // 5)
        w_shift = np.random.randint(1, fw // 5)
        tmp = np.roll(feature_map[i, :, :], h_shift, axis=0)
        train_data[i, :, :, :] = np.roll(tmp, w_shift, axis=1)
        tmp = np.roll(label, h_shift, axis=0)
        train_label[i, :, :] = np.roll(tmp, w_shift, axis=1)
    for j  in range(num):
        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        ax.imshow(train_label[j])
        plt.show()

    return train_data, train_label


def get_patch_batch_with_noise(patch, num):
    fh, fw, fc = patch.shape
    batch_data = np.zeros((num, fh, fw, fc))
    batch_data[0, :, :, :] = patch
    for i in range(1, num):
        batch_data[i, :, :, :] = patch  # add_random_noise(patch)

    return batch_data
