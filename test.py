import numpy as np
import cv2
import torch
import torchvision
from sklearn.decomposition import PCA

from config_list import config_list
from VGG16BN_user import init_vgg16
from DCF_user import DCF_layer
from add_noise import add_random_noise
from pre_process_data import get_init_patch_batch


import os
import matplotlib.pyplot as plt
from time import time

DATASET_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/dataset"
print(DATASET_PATH)
#exit()
#############################################################
def get_serach_window_size(target_h_w, scale):
    window_h_w = np.zeros(2).astype("int")
    ratio = target_h_w[0] / target_h_w[1]
    if ratio>1:
        window_h_w[0] = int(target_h_w[0] * scale)
        window_h_w[1] = int(target_h_w[0] * scale * ratio)
    else:
        window_h_w[0] = int(target_h_w[0] * scale / ratio)
        window_h_w[1] = int(target_h_w[0] * scale)

    window_h_w = window_h_w - (window_h_w % 2) + 1
    if (window_h_w[0] // 4) % 2 == 0:
        window_h_w[0] = window_h_w[0] + 4
    if (window_h_w[1] // 4) % 2 == 0:
        window_h_w[1] = window_h_w[1] + 4

    return window_h_w

def get_subwindow(im, pos, sz):
    """
    Obtain sub-window from image, with replication-padding.
    Returns sub-window of image IM centered at POS ([y, x] coordinates),
    with size SZ ([height, width]). If any pixels are outside of the image,
    they will replicate the values at the borders.
    Joao F. Henriques, 2014
    http://www.isr.uc.pt/~henriques/

    """
    h = im.shape[0]
    w = im.shape[1]
    left = pos[1] - sz[1]//2
    # if left >=0:
    #     left_padding = 0
    # else:
    #     left_padding = -left

    top = pos[0] - sz[0]//2
    # if top >=0:
    #     top_padding = 0
    # else:
    #     top_padding = -top
    right = pos[1] + sz[1]//2
    # if right <= w:
    #     right_padding = 0
    # else:
    #     right_padding = right - w

    bottom = pos[0] + sz[0] // 2
    # if bottom <= h:
    #     bottom_padding = 0
    # else:
    #     bottom_padding = bottom - h

    left_padding = max(0,-left)
    top_padding = max(0,-top)
    right_padding = max(0, right - w + 1)
    bottom_padding = max(0,bottom - h + 1)

    out = cv2.copyMakeBorder(im,top_padding,bottom_padding,left_padding,right_padding,borderType=cv2.BORDER_REPLICATE)

    #print("shi",bottom+1-top)
    out = out[top+top_padding:(bottom+top_padding+1),left+left_padding:(right+left_padding+1),:]
    return out

def gaussian_shaped_labels(sigma, sz,objSize):
    '''
    GAUSSIAN_SHAPED_LABELS
    Gaussian-shaped labels for all shifts of a sample.

    LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ)
    Creates an array of labels (regression targets) for all shifts of a
    sample of dimensions SZ. The output will have size SZ, representing
    one label for each possible shift. The labels will be Gaussian-shaped,
    with the peak at 0-shift (top-left element of the array), decaying
    as the distance increases, and wrapping around at the borders.
    The Gaussian function has spatial bandwidth SIGMA.

    Joao F. Henriques, 2014
    http://www.isr.uc.pt/~henriques/

    '''
    x = np.array(range(0,sz[0]))
    #print(x)
    y = np.array(range(0,sz[1]))
    rs, cs = np.meshgrid(y - np.ceil(sz[1] / 2),x - np.ceil(sz[0] / 2))
    #print(rs.shape)
    #print(cs.shape)
    sizeMax = np.max(objSize)
    sizeMin = np.min(objSize)
    if sizeMax / sizeMin < 1.025 and sizeMax > 120:
        alpha = 0.2
    else:
        alpha = 0.3

    labels = np.exp(-alpha * (rs** 2 / sigma[0]**2 + cs**2 / sigma[1] ** 2))
    return labels

def get_init_batch(feature_map,label):
    num, fh,fw,fc = feature_map.shape
    train_data = np.zeros((num,fh,fw,fc))
    train_label = np.zeros((num,fh,fw))
    train_data[0,:,:,:] = feature_map[0,:,:,:]
    train_label[0,:,:] = label
    for i in range(1,num):
        h_shift = np.random.randint(1,fh//5)
        w_shift = np.random.randint(1,fw//5)
        tmp = np.roll(feature_map[i,:,:], h_shift, axis=0)
        train_data[i, :, :, :] = np.roll(tmp, w_shift, axis=1)
        # ax = fig.add_subplot(111)
        # ax.imshow(test_input[0,:,:,:])
        # plt.show()
        tmp = np.roll(label, h_shift, axis=0)
        train_label[i, :, :] = np.roll(tmp, w_shift, axis=1)

    return train_data, train_label
def get_patch_batch_with_noise(patch,num):
    fh, fw, fc = patch.shape
    batch_data = np.zeros((num, fh, fw, fc))
    batch_data[0, :, :, :] = patch
    for i in range(1,num):
        batch_data[i,:,:,:] = patch# add_random_noise(patch)

    return batch_data

        




################load data############################
seq_len, gt, img_list = config_list('Dudek',DATASET_PATH)

num_channels = 3
output_sigma_factor = 0.1
cell_size=4

scale = 3
BATCH_NUM = 1

#####################################################################
result = np.zeros((seq_len, 4)).astype("int")
result[0,:] = gt[0,:]
#print(result)


im = img_list[0]

im_size = im.shape
targetLoc = gt[0,:]
# print(im[targetLoc])
# exit()
target_w_h = gt[0,2:4]
target_h_w = np.array([target_w_h[1],target_w_h[0]])
print("target_h_w:",target_h_w)
window_h_w = get_serach_window_size(target_h_w, scale)
print("window_h_w:",window_h_w)


l1_patch_num = ((window_h_w // cell_size)).astype("int")
#l1_patch_num = l1_patch_num-(l1_patch_num%2) + 1
print("l1_patch_num:",l1_patch_num)
cos_window = np.dot(np.transpose([np.hanning(l1_patch_num[0])]), [np.hanning(l1_patch_num[1])])
print("cos_window:",cos_window.shape)

sz_window = cos_window.shape

pos_h_w = np.array([targetLoc[1],targetLoc[0]]).astype("int") + np.floor(target_h_w/2).astype("int")
print(pos_h_w)

patch = get_subwindow(im, pos_h_w, window_h_w)
print("patch size:",patch.shape)
# cv2.imshow("",patch)
# cv2.waitKey(0)


loc_y = window_h_w[0]//2 - target_h_w[0]//2
loc_x = window_h_w[1]//2 - target_h_w[1]//2
loc_h = target_h_w[0]
loc_w = target_h_w[1]
patch_batch = get_init_patch_batch(patch=patch,num=BATCH_NUM,loc=[loc_y,loc_x,loc_h,loc_w],with_special_patch=True)
print("patch_batch.shape:",patch_batch.shape)


#patch  = patch.astype("uint8")
#meanImg = np.zeros(patch.shape)
#meanImg[:,:,0] = patch[:,:,0]- np.average(patch[:,:,0])
#meanImg[:,:,1] = patch[:,:,1]- np.average(patch[:,:,1])
#meanImg[:,:,2] = patch[:,:,2]- np.average(patch[:,:,2])

# print(meanImg)
# meanImg  = meanImg.astype("uint8")
# print(meanImg)

featureNet = init_vgg16().cuda()

data = patch_batch  #get_patch_batch_with_noise(patch,BATCH_NUM)
data = torch.from_numpy(data)
data = data.permute(0, 3, 1, 2)
data = data.float().cuda()
print(data.shape)
output = featureNet(data)
output=output.cpu()
output = output.permute(0, 2, 3, 1)
output = output.detach().numpy()




batch_num,hf,wf,cf=np.shape(output)
print("feature map b, h ,w, c:",batch_num,hf,wf,cf)

# if hf % 2 == 0 or wf % 2 == 0:
#     hf = hf - hf % 2 +1
#     wf = wf - wf % 2 + 1
#     output1 = np.zeros((batch_num,hf,wf,cf))
#     for i in range(batch_num):
#         output1[i,:,:,:] = cv2.resize(output[i,:,:,:],(wf,hf))
# else:
#     output1 = output
#output = output1
# _h, _w = cos_window.shape
# if output.shape != cos_window.shape:
#     output1 = np.zeros((batch_num,hf,wf,cf))
#     for i in range(batch_num):
#         output1[i,:,:,:] = cv2.resize(output[i,:,:,:],(_h, _w))
#     output = output1

batch_num,hf,wf,cf=np.shape(output)
print("feature map b, h ,w, c:",batch_num,hf,wf,cf)
#print(np.max(output),np.min(output))
window_feature = np.zeros(output.shape)
for i in range(cf):
    for j in range(batch_num):
        window_feature[j,:,:,i]  = output[j,:,:,i]  * cos_window

matrix=np.reshape(window_feature,(batch_num ,hf*wf,cf))

# pca = PCA(n_components=num_channels)
# pca = PCA(n_components= num_channels)          ####################attention!!!111
coeff = np.zeros((batch_num,hf*wf,num_channels))
for i in range(batch_num):
    pca = PCA(n_components=num_channels)
    pca.fit(matrix[i,:,:])
    coeff[i,:,:] =pca.transform(matrix[i,:,:])
print("after pca",np.shape(coeff))

#window_feature  =

# img_tmp = np.reshape(coeff,(hf,wf,-1))
# print(img_tmp.shape)
# cv2.imshow("",img_tmp)
# cv2.waitKey(0)
# np.save("feature.npy",img_tmp)

target_sz1=np.ceil(target_h_w/cell_size)

output_sigma = target_sz1*output_sigma_factor
#print(output_sigma)
label=gaussian_shaped_labels(output_sigma, l1_patch_num, target_w_h)
# cv2.imshow("",label)
# cv2.waitKey(0)
#np.save("label.npy",label)


# label1=imresize(label,[size(im1,1) size(im1,1)])*255;
# patch1=imresize(patch,[size(im1,1) size(im1,1)]);
# imd=[im1];


###########-------------------first frame initialization-----------
numEpochs=4000
featurePCA = np.reshape(coeff,(batch_num,hf,wf,-1))
np.save("feature.npy",featurePCA)



#import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(featurePCA[0])
plt.pause(5)  # 显示秒数
plt.close()
train_data, train_label  = get_init_batch(featurePCA,label)

dcf_layer = DCF_layer(train_data,target_sz1,train_label)


# fig = plt.figure()
# ax = fig.add_subplot(121)
# ax.imshow(patch)
#
# plt.pause(2)  # 显示秒数
# plt.close()

#----------------online prediction------------------
motion_sigma_factor=0.6

num_update=2          # global
cur=1
feat_update = np.zeros((num_update,1))
label_update = np.zeros((num_update,1))
target_szU = target_h_w
for  index in range(1,seq_len):
    frame = img_list[index]
    #print(frame.shape)
    #  im = imresize(im1, scale)   #  if scale!!!!

    patch = get_subwindow(frame, pos_h_w, window_h_w)
    data = patch[np.newaxis, :, :, :]
    data = torch.from_numpy(data)
    data = data.permute(0, 3, 1, 2)
    data = data.float().cuda()
    print(data.shape)
    feature = featureNet(data)
    feature = feature.cpu()
    feature = feature.permute(0, 2, 3, 1)
    feature = feature.detach().numpy()

    feature = feature[0, :, :, :]

    # hf, wf, cf = np.shape(feature)
    # if hf % 2 == 0 or wf % 2 == 0:
    #     hf = hf - hf % 2 + 1
    #     wf = wf - wf % 2 + 1
    #     feature = cv2.resize(feature, (wf, hf))
    hf, wf, cf = np.shape(feature)
    window_feature = np.zeros(feature.shape)
    for i in range(cf):
        window_feature[:, :, i] = feature[:, :, i]  * cos_window

    matrix = np.reshape(window_feature, (hf * wf, cf))

    pca = PCA(n_components=num_channels)
    pca.fit(matrix)
    coeff = pca.transform(matrix)
    featurePCA = np.reshape(coeff, (hf, wf, -1))


    # cv2.imshow("",featurePCA)
    # cv2.waitKey(10)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(featurePCA)
    # plt.show()


    predict_label = dcf_layer.search(featurePCA)
    motion_sigma = target_sz1 * motion_sigma_factor
    motion_map = gaussian_shaped_labels(motion_sigma, l1_patch_num, target_w_h)
    response = predict_label   # * motion_map
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(response)
    #plt.savefig("/home/kamata/pshow/CREST_py/test-res1/"+str(time()) + ".jpg")
    # plt.show()
    bx = fig.add_subplot(122)
    bx.imshow(patch)
    plt.pause(5)  # 显示秒数
    plt.close()

