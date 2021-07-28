#coding:utf-8

import os
import random
import shutil

old_path = '/gdata1/zhuqi/DVD/quatitive_datasets'
train_root = '/gdata1/zhuqi/DVD/train'
val_root = '/gdata1/zhuqi/DVD/val'

file_list = os.listdir(old_path)
total_num = len(file_list)  
num_val = 10

train_file_list = random.sample(file_list, total_num - num_val)

val_file_list = file_list
for i in train_file_list:
    val_file_list.remove(i) 

# train
for file in train_file_list:
    src = os.path.join(old_path, file, 'GT')
    train_gt = os.path.join(train_root, 'gt', file)
    shutil.copytree(src, train_gt)  # 此命令自带创建目标文件夹，且目标路径的最后位置为重命名的文件夹的name

    src = os.path.join(old_path, file, 'input')
    train_blur = os.path.join(train_root, 'blur', file)
    shutil.copytree(src, train_blur) 
# val
for file in val_file_list:
    src = os.path.join(old_path, file, 'GT')
    val_gt = os.path.join(val_root, 'gt', file)
    shutil.copytree(src, val_gt)  

    src = os.path.join(old_path, file, 'input')
    val_blur = os.path.join(val_root, 'blur', file)
    shutil.copytree(src, val_blur) 