import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os
from PIL import Image
import numpy as np
import random


def entropy_uncertainly(x, all_ids, model_name):
    outputs = []
    batch_num = x.shape[0]
    x1 = x.clone()
    x1 = x1.contiguous().view(batch_num, x1.shape[1], -1)

    lr_all = []

    for i in range(batch_num):
        x_t = x1[i, :, :].clone().detach()
        x_t = torch.permute(x_t, (1, 0))
        ent = -(torch.softmax(x_t, dim=1) * torch.log_softmax(x_t, dim=1))
        ent = torch.permute(ent, (1, 0))
        ent = ent.contiguous().view(ent.shape[0], x.shape[2], x.shape[3])
        outputs.append(ent)

    outputs = torch.stack(outputs, dim=0)
    for i in range(3):
        lr_all.append(torch.std(outputs[:, i+1, :, :]))

    print(lr_all)
    exit(0)

    # if not os.path.exists(model_name + "/" + "result_entroy" + "/" + str(1)):
    #     os.makedirs(model_name + "/" + "result_entroy" + "/" + str(1))
    # if not os.path.exists(model_name + "/" + "result_entroy" + "/" + str(2)):
    #     os.makedirs(model_name + "/" + "result_entroy" + "/" + str(2))
    # if not os.path.exists(model_name + "/" + "result_entroy" + "/" + str(3)):
    #     os.makedirs(model_name + "/" + "result_entroy" + "/" + str(3))
    #
    # outputs = outputs.numpy()
    # for i in range(outputs.shape[0]):
    #     img_id = all_ids[i]
    #
    #     img = outputs[i, 1, :, :] * 255
    #     im = Image.fromarray(np.uint8(img))
    #     im.save(model_name + "/" + "result_entroy" + "/" + str(1) + "/" + img_id)
    #
    #     img = outputs[i, 2, :, :] * 255
    #     im = Image.fromarray(np.uint8(img))
    #     im.save(model_name + "/" + "result_entroy" + "/" + str(2) + "/" + img_id)
    #
    #     img = outputs[i, 3, :, :] * 255
    #     im = Image.fromarray(np.uint8(img))
    #     im.save(model_name + "/" + "result_entroy" + "/" + str(3) + "/" + img_id)

    return lr_all


def lr_per_category(uncertainty_map, pred_map, cal=1, alpha=0.1):
    pred_map = (pred_map == cal)
    uncertainty = torch.masked_select(uncertainty_map, pred_map)
    lr = torch.mean(uncertainty)
    return alpha * lr.clone().detach().item() * 10


def topk_selection(feature_map, uncertainty_map, pred_map, cal_list, cal=1, k=20):
    feature_map = torch.permute(feature_map, (1, 0, 2, 3))
    feature_map = feature_map.contiguous().view(feature_map.shape[0], -1)
    feature_map = torch.permute(feature_map, (1, 0))
    uncertainty_map = uncertainty_map.contiguous().view(-1)
    pred_map = pred_map.contiguous().view(-1)

    positive_feas = []
    negative_feas = []

    for i in range(len(cal_list)):
        cls_idx = cal_list[i].item()

        pred_map_i = (pred_map.clone() == cls_idx)
        feature_map_i = feature_map[pred_map_i, :]
        uncertainty_map_i = uncertainty_map[pred_map_i]

        _, indices = torch.sort(uncertainty_map_i, dim=0)
        feature_map_i = feature_map_i[indices, :]
        
        if feature_map_i.shape[0] > k:
            feature_map_i = feature_map_i[0:k, :]
        else:
            padding_matrix = torch.Tensor(k-feature_map_i.shape[0], feature_map_i.shape[1]).cuda()
            padding_matrix = padding_matrix.copy_(feature_map_i[-1:, :])
            feature_map_i = torch.cat((feature_map_i, padding_matrix), dim=0)
            
        if cls_idx == cal:
            positive_feas.append(feature_map_i)
        else:
            negative_feas.append(feature_map_i)

    # positive_feas = positive_feas[0]
    positive_labs = torch.ones(k).cuda()

    # negative_feas = torch.stack(negative_feas, dim=0)
    # negative_feas = negative_feas.contiguous().view(negative_feas.shape[0] * negative_feas.shape[1], negative_feas.shape[2])
    negative_labs = torch.zeros(k).cuda()

    # feas = torch.cat((positive_feas, negative_feas), dim=0)
    # labs = torch.cat((positive_labs, negative_labs), dim=0)

    return positive_feas, negative_feas, positive_labs, negative_labs







