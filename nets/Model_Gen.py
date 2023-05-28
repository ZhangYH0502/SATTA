import torch
import torch.nn as nn
import numpy as np
from nets.UNets import UNet_Encoder, UNet_Decoder
from nets.TransformerLayers import TransformerEncoderLayer, TransformerDecoderLayer, positional_encoding_2d, LabelEmbedding


class OurModel(nn.Module):
    def __init__(self, num_labels=4):
        super(OurModel, self).__init__()

        # UNet Encoder: [N, 1024, 32, 32]
        self.unet_encoder = UNet_Encoder(in_channel=1)
        self.unet_decoder = UNet_Decoder(out_channel=num_labels)

    def forward(self, images):
        features = self.unet_encoder(images)
        mask_pred = self.unet_decoder(features)

        return mask_pred, features


class OurModel_Tran(nn.Module):
    def __init__(self, num_labels=4, layers=3, heads=4, dropout=0.1):
        super(OurModel_Tran, self).__init__()

        # UNet Encoder: [N, 1024, 32, 32]
        self.unet_encoder = UNet_Encoder(in_channel=1)
        self.hidden = 1024

        # UNet Decoder
        self.unet_decoder = UNet_Decoder(out_channel=num_labels)

        # self.InstanceNorm = nn.InstanceNorm2d(num_features=1024, affine=True)

        # Transformer
        self.class_embedding = LabelEmbedding(num_labels=3, hidden=self.hidden)
        self.position_encoding = positional_encoding_2d(self.hidden, 32, 32).unsqueeze(0)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(self.hidden, heads, dropout) for _ in range(layers)])
        # self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(hidden, heads, dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.output_linear1 = torch.nn.Linear(self.hidden, num_labels-1)

        # Other
        self.LayerNorm = nn.LayerNorm(self.hidden)

    def forward(self, images):

        features = self.unet_encoder(images)

        pos_encoding = self.position_encoding

        features = features + pos_encoding.cuda()

        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

        initial_class_embedding = self.class_embedding(features.size(0))

        embeddings = torch.cat((features, initial_class_embedding), 1)
        embeddings = self.LayerNorm(embeddings)

        for layer in self.encoder_layers:
            embeddings = layer(embeddings)

        updated_fearures = embeddings[:, 0:features.size(1), :]
        updated_fearures = updated_fearures.permute(0, 2, 1)
        updated_fearures = updated_fearures.view(updated_fearures.size(0), updated_fearures.size(1), 32, 32)
        mask_pred = self.unet_decoder(updated_fearures)

        class_embedding = embeddings[:, -initial_class_embedding.size(1):, :]
        class_pred = self.output_linear1(class_embedding)
        diag_mask = torch.eye(class_pred.size(1)).unsqueeze(0).repeat(class_pred.size(0), 1, 1).cuda()
        class_pred = (class_pred * diag_mask).sum(-1)

        return mask_pred, class_pred
