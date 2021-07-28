#coding:utf-8

import os
import random
import shutil


train_blur = '/gdata1/zhuqi/DVD/train/blur'
train_blur_depth = '/gdata1/zhuqi/DVD/train/blur_depth'
train_gt = '/gdata1/zhuqi/DVD/train/gt'
train_gt_depth = '/gdata1/zhuqi/DVD/train/gt_depth'

val_blur = '/gdata1/zhuqi/DVD/val/blur'
val_blur_depth = '/gdata1/zhuqi/DVD/val/blur_depth'
val_gt = '/gdata1/zhuqi/DVD/val/gt'
val_gt_depth = '/gdata1/zhuqi/DVD/val/gt_depth'

# train
# file_list = os.listdir(train_blur)
# for file in file_list:
#     i = os.path.join(train_blur, file)
#     output = os.path.join(train_blur_depth, file)
#     os.mkdir(output)
#     os.system('python run.py --model_type dpt_large -i ' + i + ' -o ' + output)

file_list = os.listdir(train_gt)
for file in file_list:
    i = os.path.join(train_gt, file)
    output = os.path.join(train_gt_depth, file)
    os.mkdir(output)
    os.system('python run.py --model_type dpt_large -i ' + i + ' -o ' + output)

# file_list = os.listdir(val_blur)
# for file in file_list:
#     i = os.path.join(val_blur, file)
#     output = os.path.join(val_blur_depth, file)
#     os.mkdir(output)
#     os.system('python run.py --model_type dpt_large -i ' + i + ' -o ' + output)

# file_list = os.listdir(val_gt)
# for file in file_list:
#     i = os.path.join(val_gt, file)
#     output = os.path.join(val_gt_depth, file)
#     os.mkdir(output)
#     os.system('python run.py --model_type dpt_large -i ' + i + ' -o ' + output)

#python run.py --model_type dpt_large -i /gdata1/zhuqi/DVD/val/blur/GOPR9643 -o /gdata1/zhuqi/DVD/val/blur_depth/GOPR9643