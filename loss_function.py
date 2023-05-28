import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.Loss = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        result = self.Loss(predict, target)
        return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-5, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, ignore_index=0):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.b_dice = BinaryDiceLoss()

    def forward(self, predict, target):
        num_labels = predict.shape[1]
        predict = torch.softmax(predict, dim=1)
        total_loss = []

        for i in range(num_labels):
            if i != self.ignore_index:
                binary_dice_loss = self.b_dice(predict[:, i, :, :].clone(), target.clone() == i)
                total_loss.append(binary_dice_loss)

        total_loss = torch.stack(total_loss, dim=0)

        return total_loss.mean()


class CE_Dice_Loss(nn.Module):
    def __init__(self):
        super(CE_Dice_Loss, self).__init__()
        self.CE_Loss = CrossEntropyLoss()
        self.Dice_Loss = DiceLoss()

    def forward(self, predict, target):
        result = self.CE_Loss(predict.clone(), target.clone()) + self.Dice_Loss(predict.clone(), target.clone())
        return result


class ProxyPLoss(nn.Module):
    # self.proxycloss = ProxyPLoss()
    def __init__(self, num_classes=4, scale=12, train=True):
        super(ProxyPLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.LongTensor([i for i in range(num_classes)])
        self.scale = scale
        self.train = train

    def forward(self, feature, target, proxy):
        feature = torch.permute(feature, (1, 0, 2, 3))
        feature = feature.contiguous().view(feature.shape[0], -1)
        feature = torch.permute(feature, (1, 0))

        target = target.contiguous().view(-1)

        if self.train:
            num = feature.shape[0]
            idx_list = np.arange(num, dtype=int)
            random.shuffle(idx_list)
            idx_list = idx_list[0:100]
            feature = feature[idx_list, :]
            target = target[idx_list]

        feature = F.normalize(feature, p=2, dim=1)
        pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))  # (N, C)

        label = (self.label.unsqueeze(1).to(feature.device) == target.unsqueeze(0)) # (C, N)

        pred = torch.masked_select(pred.transpose(1, 0), label)  # N,

        pred = pred.unsqueeze(1)  # (N, 1)

        feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # (N, N)

        feature = feature * ~label_matrix  # get negative matrix
        feature = feature.masked_fill(feature < 1e-6, -np.inf)  # (N, N)

        logits = torch.cat([pred, feature], dim=1)  # (N, 1+N)
        label = torch.zeros(logits.size(0), dtype=torch.long).to(feature.device)
        loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)

        return loss


class ProxyPLoss_Single(nn.Module):
    def __init__(self, num_classes=2, scale=12):
        super(ProxyPLoss_Single, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.LongTensor([i for i in range(num_classes)])
        self.scale = scale

    def forward(self, feature, target, proxy, cls):
        feature = F.normalize(feature, p=2, dim=1)
        pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))  # (N, C)

        pred_p = pred[:, cls:cls+1].clone()
        pred_n = pred[:, torch.arange(pred.size(1)) != cls].clone()
        pred_n, _ = torch.max(pred_n, dim=1, keepdim=True)

        pred = torch.cat((pred_p, pred_n), dim=1)

        label = (self.label.unsqueeze(1).to(feature.device) == target.unsqueeze(0)) # (C, N)

        pred = torch.masked_select(pred.transpose(1, 0), label)  # N,

        pred = pred.unsqueeze(1)  # (N, 1)

        feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # (N, N)

        feature = feature * ~label_matrix  # get negative matrix
        feature = feature.masked_fill(feature < 1e-6, -np.inf)  # (N, N)

        logits = torch.cat([pred, feature], dim=1)  # (N, 1+N)
        label = torch.zeros(logits.size(0), dtype=torch.long).to(feature.device)
        loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)

        return loss


if __name__ == "__main__":

    fc_proj = nn.Parameter(torch.FloatTensor(2, 3))
    print(fc_proj)
