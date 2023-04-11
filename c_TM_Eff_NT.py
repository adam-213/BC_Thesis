import random

import torch
import torch.nn as nn
import timm
import math
import numpy as np

from matplotlib import pyplot as plt


def _init_layer(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)


class PoseEstimationModel(nn.Module):
    def __init__(self, num_channels=7):
        super(PoseEstimationModel, self).__init__()
        self.workround = False

        self.Wloss = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.backbone = timm.create_model("resnetv2_50d", pretrained=False, in_chans=num_channels)
        self.backbone.classifier = nn.Identity()

        self.W_head = nn.Sequential(
            nn.Linear(1000 + 3, 8192),
            nn.BatchNorm1d(8192),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.5),
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.15),
            nn.Dropout(0.3),
            nn.Linear(2048, 512 + 128),
            nn.BatchNorm1d(512 + 128),
            nn.LeakyReLU(0.15),
            nn.Dropout(0.2),
            nn.Linear(512 + 128, 256 + 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(256 + 64, 3)
        )

        self.W_head.apply(_init_layer)
        self.backbone.apply(_init_layer)

    def forward(self, x, XYZ):
        if x.shape[0] == 1:
            # workround for batch size 1, as batchnorm does not work with batch size 1
            x = torch.cat([x, x], dim=0)
            XYZ = torch.cat([XYZ, XYZ], dim=0)
            self.workround = True
        try:
            features = self.backbone(x)
        except RuntimeError as e:
            print(e)
            print(x.shape, "X")
            return None

        stacked = torch.cat((features, XYZ), dim=1)
        try:
            W = self.W_head(stacked)
        except RuntimeError as e:
            print(e)
            print(stacked.shape, "W")
            return None
        return W

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

                Img = Img[:,:1, :, :]

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
            print("wtf",e)

        if self.workround:
            hat_W = hat_W[0]
            gt_W = gt_W[0]
            self.workround = False

        wloss = self.Wloss(hat_W, gt_W)
        return wloss
