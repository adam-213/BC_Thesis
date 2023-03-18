import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()

        self.conv_mask = nn.Conv2d(1, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_feat = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask):
        mask = self.conv_mask(mask)
        feat = self.conv_feat(x)

        mask = F.softmax(mask, dim=-1)
        feat = feat * mask

        feat = self.conv_out(feat)
        feat = self.gamma * feat + x

        return feat


class TM_CNN(nn.Module):
    def __init__(self, in_channels, attention_channels):
        super(TM_CNN, self).__init__()
        # +1 for mask channel
        self.conv1 = nn.Conv2d(in_channels + 1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.attention = Attention(256, attention_channels)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 16)

    def forward(self, x, mask):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention(x, mask)
        x = x.view(-1, 256 * 8 * 8)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
