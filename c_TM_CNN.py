import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, instances):
        super(Attention, self).__init__()

        self.conv_mask = nn.Conv2d(instances, out_channels * instances, kernel_size=1, stride=1, padding=0,
                                   groups=instances)
        self.conv_feat = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_channels = out_channels

    def forward(self, x, masks):
        batch_size, instances, _, _ = masks.size()
        feat = self.conv_feat(x)
        out = torch.zeros_like(feat)

        masks = self.conv_mask(masks)
        masks = F.softmax(masks.view(batch_size, instances, -1), dim=-1).view_as(masks)

        for i in range(instances):
            mask = masks[:, i * self.out_channels:(i + 1) * self.out_channels, :, :]
            instance_feat = feat * mask
            instance_feat = self.conv_out(instance_feat)
            out += instance_feat

        out = self.gamma * out + x
        return out


class TM_CNN(nn.Module):
    def __init__(self, in_channels, attention_channels, instances):
        super(TM_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.attention = Attention(64, 64, instances)
        self.fc1 = nn.Linear(792576, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 16)
        self.maxpoll = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, masks):
        masks = self.reshape_masks(masks)  # reshape to 1/4 of the original resolution
        x = F.relu(self.conv1(x))
        x = self.maxpoll(x)
        x = F.relu(self.conv2(x))
        x = self.maxpoll(x)
        x = F.relu(self.conv3(x))
        x = self.maxpoll(x)
        x = self.attention(x, masks)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reshape_masks(self, masks):
        # reshape to 1/4 of the original resolution
        batch_size, instances, height, width = masks.size()
        masks = F.interpolate(masks, size=(height // 8, width // 8), mode='nearest')
        return masks

    def loss(self, pred, gt):
        # reshape
        pred = pred.view(-1, 4, 4)  # (batch_size, 4, 4)
        # split the matrix into rotation and translation
        pred_r = pred[:, :3, :3]
        pred_t = pred[:, :3, 3]
        gt_r = gt[:, :3, :3]
        gt_t = gt[:, :3, 3]

        # compute the loss
        # translation loss is just classic mse
        loss_t = F.mse_loss(pred_t, gt_t)
        # rotation loss is geodesic distance + frobenius norm
        loss_r = torch.norm(pred_r - gt_r, p='fro', dim=(1, 2)) + torch.norm(pred_r - gt_r, p=2, dim=(1, 2))
        loss_r = torch.mean(loss_r)
        return loss_t + loss_r
