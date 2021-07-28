import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import os
import random
from PIL import Image,ImageOps
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_patch(img_nn, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_nn[0].size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)


    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy, ix, iy + ip, ix + ip)) for j in img_nn]  # [:, iy:iy + ip, ix:ix + ip]


    return  img_nn, img_tar


def resize(img_nn, img_tar):
    (ih, iw) = img_nn[0].size
    (th, tw) = (ih, iw)

    while iw % 8 != 0:
        tw += 1
    while ih % 8 != 0:
        th += 1

    img_tar = img_tar.resize((th,tw))  # [:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.resize((th,tw)) for j in img_nn]  # [:, iy:iy + ip, ix:ix + ip]

    return  img_nn, img_tar


def augment(img_nn, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return  img_nn, img_tar

def get_image(img):
    img = Image.open(img).convert('RGB')
    return img


def load_image_train2(group):
    images = [get_image(img) for img in group]
    inputs = images[:-1]
    target = images[-1]
    # if black_edges_crop == True:
    #     inputs = [indiInput[70:470, :, :] for indiInput in inputs]
    #     target = target[280:1880, :, :]
    #     return inputs, target
    # else:
    return inputs, target


def transform():
    return Compose([
        ToTensor(),
    ])


class DatasetFromFolder(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, upscale_factor, data_augmentation, group_file, patch_size, black_edges_crop, hflip, rot, transform=transform()):
        super(DatasetFromFolder, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(group_file))]
        # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'

        print(np.array(groups).shape)  # 3000
        print(groups[0])

        self.image_filenames = [group.split('|') for group in groups]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size
        self.black_edges_crop = black_edges_crop
        self.hflip = hflip
        self.rot = rot

        print(self.image_filenames[0])

    def __getitem__(self, index):

        inputs, target = load_image_train2(self.image_filenames[index])  

        if self.patch_size != None:
            inputs, target = get_patch(inputs, target, self.patch_size, self.upscale_factor)

        else:
            inputs, target = resize(inputs, target)

        if self.data_augmentation:
            inputs, target = augment(inputs, target, self.hflip, self.rot)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                             torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                             torch.unsqueeze(inputs[4], 0)))


        return inputs, target

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    output = 'visualize'
    if not os.path.exists(output):
        os.mkdir(output)
    dataset = DatasetFromFolder(4, True, '/gdata1/zhuqi/REDS/val/meta_info_REDSval_official_test_GT.txt', 64, True, True, True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=False)
    for i, (inputs, target) in enumerate(dataloader):
        if i > 10:
            break
        if not os.path.exists(os.path.join(output, 'group{}'.format(i))):
            os.mkdir(os.path.join(output, 'group{}'.format(i)))
        input0, input1, input2, input3, input4 = inputs[0][0], inputs[0][1], inputs[0][2], inputs[0][3], inputs[0][4]
        vutils.save_image(input0, os.path.join(output, 'group{}'.format(i), 'input0.png'))
        vutils.save_image(input1, os.path.join(output, 'group{}'.format(i), 'input1.png'))
        vutils.save_image(input2, os.path.join(output, 'group{}'.format(i), 'input2.png'))
        vutils.save_image(input3, os.path.join(output, 'group{}'.format(i), 'input3.png'))
        vutils.save_image(input4, os.path.join(output, 'group{}'.format(i), 'input4.png'))
        vutils.save_image(target, os.path.join(output, 'group{}'.format(i), 'target.png'))