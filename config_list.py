import numpy as np
import cv2
from os import listdir
import os
def load_img_data(dir):
    # DATA_DIR = "/media/pshow/0EB217730EB21773/VOT_full/KCF_traker/bag/"


    filename_list=listdir(dir)

    filename_list=sorted(filename_list)

    jpg_img_list=[]

    #可以同过简单后缀名判断，筛选出你所需要的文件(这里以.jpg为例)
    for filename in filename_list:#依次读入列表中的内容
        if filename.endswith('jpg'):# 后缀名'jpg'匹对
            im = cv2.imread(dir+filename,-1)
            try:
                if im.shape[2] != 3:
                    im= cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            except:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            jpg_img_list.append(im)


    print("list length: ",len(jpg_img_list))

    return len(jpg_img_list), jpg_img_list


def config_list(seq_name, dataset_dir):
    gt_dir =  dataset_dir + '/' + seq_name + '/groundtruth_rect.txt'
    try:
        gt = np.loadtxt(gt_dir)
    except:
        gt = np.loadtxt(gt_dir,delimiter=',')
    img_dir = dataset_dir + '/' + seq_name + '/img/'
    seq_len, img_list = load_img_data(img_dir)


    return seq_len, gt.astype("int"), img_list






# seq_len, gt, img_list = config_list('Bird1',os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/dataset")
# index = 0
# #print(img_list)
# while(index < seq_len):
#     frame = img_list[index]
#     #print(frame)
#     cv2.rectangle(frame,(gt[index][0], gt[index][1]), (gt[index][0]+gt[index][2], gt[index][1]+gt[index][3]),
#                           (0, 255, 255), 1)
#     cv2.imshow('tracking', frame)
#     cv2.waitKey(100)
#     index +=1
# cv2.destroyAllWindows()