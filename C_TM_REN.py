import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import numpy as np

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class HConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(HConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Apply standard 2D convolution
        conv_output = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

        # Compute Discrete Fourier Transform (DFT) along the channel axis
        dft_output = torch.fft.fft(conv_output, dim=1)

        # Compute inverse DFT to get rotation-equivariant output
        rotation_equivariant_output = torch.fft.ifft(dft_output, dim=1).real

        return rotation_equivariant_output


def _init_layer(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)


class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # Normalize the predicted vectors
        pred_norm = F.normalize(pred, p=2, dim=1)

        cos_sim_loss = 1 - self.cosine_similarity(pred_norm, target)
        l2_loss = self.mse_loss(pred_norm, target)

        # Weighted combination of the cosine similarity and L2 losses
        combined_loss = self.alpha * cos_sim_loss + (1 - self.alpha) * l2_loss
        return torch.mean(combined_loss)


class RENBackbone(nn.Module):
    def __init__(self, in_channels):
        super(RENBackbone, self).__init__()

        self.harmonic_conv1 = HConv2d(in_channels, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, padding=1)

        # Define the REN block
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Sequential(
            HConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            HConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        ))

        for _ in range(1, num_blocks):
            layers.append(nn.Sequential(
                HConv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                HConv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels)
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.harmonic_conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class REN(nn.Module):
    def __init__(self, in_channels):
        super(REN, self).__init__()
        self.backbone = RENBackbone(in_channels)
        self.W_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.15),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.15),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )
        self.Wloss = CustomLoss()
        self.W_head.apply(_init_layer)
        self.workaround = False

    def forward(self, x, XYZ):
        if x.shape[0] == 1:
            x = torch.cat([x, x], dim=0)
            self.workaround = True
        x = self.backbone(x)
        x = self.W_head(x)
        return x

    def loss_W(self, hat_W, gt_W, img=None, XYZ=None, batchepoch=(0, 0)):

        magnitude = torch.sqrt(torch.sum(hat_W ** 2, dim=1)).view(-1, 1)

        hat_W = hat_W / magnitude
        try:
            if random.random() < 0.005:
                hat_w = hat_W.cpu()
                gt_w = gt_W.cpu()
                img = img.cpu()
                XYZ = XYZ.cpu()

                hat_W = hat_w.clone().cuda()
                gt_W = gt_w.clone().cuda()

                hat_w = hat_w.detach().numpy()
                gt_w = gt_w.detach().numpy()
                Img = img.detach().numpy()
                XYZ = XYZ.detach().numpy()
                for i in range(hat_w.shape[0]):
                    print(hat_w[i].tolist(), gt_w[i].tolist())

                Img = Img[:, :1, :, :]

                for img_index in range(Img.shape[0]):
                    cur_img = Img[img_index]
                    cur_img = np.transpose(cur_img, (1, 2, 0))

                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(cur_img)

                    start = np.array([cur_img.shape[1] // 2, cur_img.shape[0] // 2])

                    scaling_factor = 100
                    end_hat_w = start + scaling_factor * np.array([hat_w[img_index][0], hat_w[img_index][1]])
                    end_gt_w = start + scaling_factor * np.array([gt_w[img_index][0], gt_w[img_index][1]])

                    # Create a set of points along the line for hat_w and gt_w
                    num_points = 100
                    points_hat_w = np.linspace(start, end_hat_w, num_points)
                    points_gt_w = np.linspace(start, end_gt_w, num_points)

                    # Plot hat_w line in green
                    for i in range(num_points - 1):
                        ax.plot(points_hat_w[i:i + 2, 0], points_hat_w[i:i + 2, 1], '_', color='g', alpha=0.2)

                    # Plot gt_w line in blue
                    for i in range(num_points - 1):
                        ax.plot(points_gt_w[i:i + 2, 0], points_gt_w[i:i + 2, 1], '_', color='b', alpha=0.2)

                    fig.savefig(f"xx_hn_d{batchepoch[1]}_{batchepoch[0]}_{img_index}.png")
                    plt.close(fig)
        except Exception as e:
            print("wtf", e)

        if self.workaround:
            hat_W = hat_W[:1]
            gt_W = gt_W[:1]
            self.workaround = False

        wloss = self.Wloss(hat_W, gt_W)
        return wloss
