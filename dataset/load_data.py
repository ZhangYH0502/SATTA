import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import warnings
import numpy as np
import random
from PIL import Image
from dataset.fluid_dataset import FluidDataset

warnings.filterwarnings("ignore")


def get_data(args):

    test_dataset = None
    test_loader = None

    data_root = os.path.join(args.data_root, 'FluidSegDataset')

    split_data_train = np.load('dataset/data_list_train.npz')
    cirrus_list_train = split_data_train['cirrus_list'].tolist()
    spectralis_list_train = split_data_train['spectralis_list'].tolist()
    topcon1_list_train = split_data_train['topcon1_list'].tolist()
    topcon2_list_train = split_data_train['topcon2_list'].tolist()
    topcon_list_train = topcon1_list_train + topcon2_list_train

    split_data_test = np.load('dataset/data_list_test.npz')
    cirrus_list_test = split_data_test['cirrus_list'].tolist()
    spectralis_list_test = split_data_test['spectralis_list'].tolist()
    topcon_list_test = split_data_test['topcon_list'].tolist()

    # cirrus_list = cirrus_list_train + cirrus_list_test
    # spectralis_list = spectralis_list_train + spectralis_list_test
    # topcon_list = topcon_list_train + topcon_list_test

    train_list = topcon_list_train + cirrus_list_train
    valid_list = spectralis_list_test

    train_dataset = FluidDataset(img_root=data_root, data_list=train_list, flg="train")
    valid_dataset = FluidDataset(img_root=data_root, data_list=valid_list, flg="valid")

    if train_dataset is not None:
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers,
                                  drop_last=False)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=args.workers)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=args.workers)

    return train_loader, valid_loader, test_loader


class GetDataFromAnotherDomain(nn.Module):

    def __init__(self, args, dataset="cirrus"):
        super(GetDataFromAnotherDomain, self).__init__()

        self.data_root = os.path.join(args.data_root, 'FluidSegDataset')
        self.dataset = dataset
        self.batch_size = args.batch_size

        split_data_train = np.load('dataset/data_list_train.npz')
        if self.dataset == "cirrus":
            self.data_list = split_data_train['cirrus_list'].tolist()
        elif self.dataset == "topcon":
            self.data_list = split_data_train['topcon_list'].tolist()
        else:
            self.data_list = split_data_train['spectralis_list'].tolist()

    def forward(self):

        idx_list = np.arange(len(self.data_list), dtype=int)
        random.shuffle(idx_list)
        idx_list = idx_list[:self.batch_size]

        images_batch = []
        masks_batch = []
        imageID_batch = []

        for i in range(self.batch_size):
            image_id = self.data_list[idx_list[i]]
            imageID_batch.append(image_id)

            images = Image.open(self.data_root + "/" + "images" + "/" + image_id)
            masks = Image.open(self.data_root + "/" + "labels" + "/" + image_id)

            images = torch.Tensor(np.array(images))
            images = (images - torch.min(images)) / (torch.max(images) - torch.min(images))
            images = images.unsqueeze(0)

            masks = torch.LongTensor(np.array(masks))

            images_batch.append(images)
            masks_batch.append(masks)

        images_batch = torch.stack(images_batch, dim=0)
        masks_batch = torch.stack(masks_batch, dim=0)

        sample = {}
        sample['images'] = images_batch
        sample['masks'] = masks_batch
        sample['imageIDs'] = imageID_batch

        return sample
