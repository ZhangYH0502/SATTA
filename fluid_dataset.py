import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random


class FluidDataset(Dataset):

    def __init__(self, img_root, data_list, flg, domain_num=2):

        self.flg = flg

        if self.flg == "train":
            self.root = img_root + "/" + "train"
        else:
            self.root = img_root + "/" + "test"

        self.data_list = data_list
        self.domain_num = domain_num

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id = self.data_list[idx]

        images = Image.open(self.root + "/" + "images" + "/" + image_id)
        masks = Image.open(self.root + "/" + "labels" + "/" + image_id)

        # if self.flg == "train":
        #     random_rot = random.uniform(-15, 15)
        #     images = images.rotate(random_rot)
        #     labels = labels.rotate(random_rot)
        #
        #     rand_flip = random.randint(0, 1)
        #     if rand_flip == 1:
        #         images = images.transpose(Image.FLIP_LEFT_RIGHT)
        #         labels = labels.transpose(Image.FLIP_LEFT_RIGHT)

        images = torch.Tensor(np.array(images))
        images = (images - torch.min(images)) / (torch.max(images) - torch.min(images))
        images = images.unsqueeze(0)

        masks = torch.LongTensor(np.array(masks))

        # class_labels = torch.zeros(3)
        # if 1 in masks:
        #     class_labels[0] = 1
        # if 2 in masks:
        #     class_labels[1] = 1
        # if 3 in masks:
        #     class_labels[2] = 1

        sample = {}
        sample['images'] = images[:, 0:512:2, 0:512:2]
        sample['masks'] = masks[0:512:2, 0:512:2]
        # sample['class_labels'] = class_labels
        sample['imageIDs'] = image_id

        return sample
