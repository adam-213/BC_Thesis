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
        self.out_channels = out_channels

    def forward(self, x, masks):
        batch_size, _, _ = masks.size()
        feat = self.conv_feat(x)
        out = torch.zeros_like(feat)

        masks = self.conv_mask(masks.unsqueeze(1))
        masks = F.softmax(masks.view(batch_size, -1), dim=-1).view_as(masks)

        instance_feat = feat * masks
        instance_feat = self.conv_out(instance_feat)
        out += instance_feat

        out = self.gamma * out + x
        return out


class TM_CNN(nn.Module):
    def __init__(self, in_channels, attention_channels):
        super(TM_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 24, kernel_size=3, stride=1, padding=1)
        self.attention = Attention(24, 24)
        self.fc1 = nn.Linear(int(792576 // 2 // 2 * 1.5), 512)
        #self.fc2 = nn.Linear(8192, 1024)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 16)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, masks):
        masks = self.reshape_masks(masks)
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.attention(x, masks)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def reshape_masks(self, masks):
        masks.unsqueeze_(1)
        batch_size, channels, height, width = masks.size()
        masks = F.interpolate(masks, size=(height // 8, width // 8), mode='nearest')
        return masks.squeeze(1)

    def loss(self, pred, gt):
        pred = pred.view(-1, 4, 4)
        pred_r = pred[:, :3, :3]
        pred_t = pred[:, :3, 3]
        gt_r = gt[:, :3, :3]
        gt_t = gt[:, :3, 3]

        loss_t = F.mse_loss(pred_t, gt_t)
        loss_r = torch.norm(pred_r - gt_r, p='fro', dim=(1, 2)) + torch.norm(pred_r - gt_r, p=2, dim=(1, 2))
        loss_r = torch.mean(loss_r)
        return loss_t + loss_r
