import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights


class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers=1):
        super(CustomTransformerEncoder, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers)

    def forward(self, src):
        return self.transformer_encoder(src)


class TM_CNN(nn.Module):
    def __init__(self, in_channels, d_model, nhead, num_layers=1):
        super(TM_CNN, self).__init__()
        self.resnet = resnet34()
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.transformer = CustomTransformerEncoder(d_model, nhead, num_layers)

        self.fc1 = nn.Linear(422400, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 192)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(192, 16)

    def forward(self, x):
        x = self.resnet(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        # x = x.view(b, -1)
        x = x.reshape(b, -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

    def loss(self, pred, gt):
        pred = pred.view(-1, 4, 4)
        pred_r = pred[:, :3, :3]
        pred_t = pred[:, 3, :3]
        gt_r = gt[:, :3, :3]
        gt_t = gt[:, 3, :3]

        loss_t = F.mse_loss(pred_t, gt_t)

        # Geodesic loss for rotation
        R = pred_r.bmm(gt_r.transpose(1, 2))
        trace = torch.diagonal(R, dim1=1, dim2=2).sum(dim=1)
        trace = torch.clamp(trace, min=-1.0 + 1e-7, max=1.0 - 1e-7)  # Clip trace to a valid range
        loss_r = torch.mean(torch.arccos((trace - 1) / 2))

        # Penalty for non-zero elements in specific positions
        zero_positions = pred[:, :3, 3]
        penalty = torch.mean(zero_positions ** 2)

        if random.random() < 0.01:
            print("gt", gt[0])
            print("pred", pred[0])
        # Weights for translation, rotation, and penalty
        w_t = 1
        w_r = 300
        w_p = 1

        total_loss = w_t * loss_t + w_r * loss_r + 0 * penalty

        return total_loss
