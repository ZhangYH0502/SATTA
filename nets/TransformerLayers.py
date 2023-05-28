import torch
import torch.nn as nn
import math
import numpy as np


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1, dim_feedforward=2048, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        # Custom method to return attn outputs. Otherwise same as nn.TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dropout=0.1, dim_feedforward=2048, activation="relu"):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,  memory_key_padding_mask=None):

        tgt2, sim_mat_1 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, sim_mat_2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def positional_encoding_1d(d_model, max_len):
    # Compute the positional encodings once in log space.
    pe = torch.zeros(d_model, max_len)
    position = torch.arange(0., max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

    pe[0::2, :] = torch.sin(position * div_term).transpose(0, 1)
    pe[1::2, :] = torch.cos(position * div_term).transpose(0, 1)

    return pe


def positional_encoding_2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class LabelEmbedding(nn.Module):
    """
    This is the label embedding layer
    """
    def __init__(self, num_labels=3, hidden=1024):
        super().__init__()
        self.num_labels = num_labels
        self.hidden = hidden

        self.label_input = torch.Tensor(np.arange(self.num_labels)).view(1, -1).long()
        self.label_lt = torch.nn.Embedding(self.num_labels, self.hidden, padding_idx=None)

    def forward(self, batch_num):
        const_label_input = self.label_input.repeat(batch_num, 1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)
        # print('init_label_embeddings: ', init_label_embeddings.shape)
        return init_label_embeddings
