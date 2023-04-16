import random

import torch
import torch.nn as nn
import timm
import math
import numpy as np
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt


def _init_layer(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)


class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5, lambda_magnitude=0.1, lambda_range=1.0, use_geodesic_loss=True):
        super().__init__()
        self.alpha = alpha
        self.lambda_magnitude = lambda_magnitude
        self.lambda_range = lambda_range
        self.use_geodesic_loss = use_geodesic_loss

    def geodesic_loss(self, u, v):
        dot_product = torch.sum(u * v, dim=1)
        norm_u = torch.norm(u, p=2, dim=1)
        norm_v = torch.norm(v, p=2, dim=1)
        angle_cosine = dot_product / (norm_u * norm_v)
        angle_cosine = torch.clamp(angle_cosine, -1.0, 1.0)
        return torch.acos(angle_cosine)

    def squared_euclidean_loss(self, u, v):
        return torch.sum((u - v) ** 2, dim=1)

    def magnitude_regularization(self, u):
        return torch.abs(torch.norm(u, p=2, dim=1) - 1)

    def range_regularization(self, u):
        return torch.sum(F.relu(torch.abs(u) - 1), dim=1)

    def forward(self, pred, target, index=None):
        # Normalize the predicted vectors
        pred_norm = F.normalize(pred, p=2, dim=1)

        # Calculate the main loss (geodesic or squared Euclidean)
        if self.use_geodesic_loss:
            main_loss = self.geodesic_loss(pred_norm, target)
        else:
            main_loss = self.squared_euclidean_loss(pred_norm, target)

        # Calculate the magnitude regularization term
        mag_reg = self.magnitude_regularization(pred_norm)

        # Calculate the range regularization term
        range_reg = self.range_regularization(pred_norm)

        # Weighted combination of the main loss, magnitude regularization, and range regularization
        combined_loss = self.alpha * main_loss + self.lambda_magnitude * mag_reg + self.lambda_range * range_reg

        if index is not None:
            # Return the loss for a single instance for plotting
            return combined_loss[index]
        else:
            return torch.mean(combined_loss)


class RotationEquivariantLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_rotations=8, kernel_size=3, stride=1, padding=1):
        super(RotationEquivariantLayer, self).__init__()
        self.num_rotations = num_rotations
        self.conv = nn.Conv2d(in_channels, out_channels * num_rotations, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels * num_rotations)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class PoseEstimationModel(nn.Module):
    def __init__(self, num_channels=7, mid_channels=32, num_rotations=12):
        super(PoseEstimationModel, self).__init__()
        self.workround = False

        self.Wloss = CustomLoss(use_geodesic_loss=True)
        self.input_norm = nn.BatchNorm2d(num_channels)
        # self.rotation_equivariant_layer = RotationEquivariantLayer(num_channels, mid_channels, num_rotations)
        from transformers import EfficientFormerConfig, EfficientFormerModel, AutoImageProcessor

        # Initializing a EfficientFormer efficientformer-l1 style configuration
        configuration = EfficientFormerConfig(num_channels=num_channels)

        # Initializing a EfficientFormerModel (with random weights) from the efficientformer-l3 style configuration
        model = EfficientFormerModel(configuration)
        processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
        self.processor = processor

        self.backbone = model
        # self.backbone.classifier = nn.Identity()

        self.W_head = nn.Sequential(
            nn.Linear(1000 + 3, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(1024, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.15),
            nn.Dropout(0.2),
            nn.Linear(384, 160),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(160, 3)
        )

        self.W_head.apply(_init_layer)
        self.backbone.apply(_init_layer)

    def forward(self, x, XYZ):
        x = self.processor(x, return_tensors="pt").to(self.device)
        try:
            print(x.shape)
            x = self.layer(self.input_norm, x)
            backbonex = self.layer(self.backbone, x)
            x = torch.cat([backbonex, XYZ], dim=1)
            x = self.layer(self.W_head, x, XYZ)
            return x
        except OverflowError as e:
            print(e)
            return None

    def loss_W(self, hat_W, gt_W, *plotargs):
        # normalisation, but im not sure if its helping or hurting
        # magnitude = torch.sqrt(torch.sum(hat_W ** 2, dim=1)).view(-1, 1)
        # hat_W = hat_W / magnitude
        if random.random() < 10.001: self.plot(hat_W, gt_W, *plotargs)

        if self.workround:
            hat_W = hat_W[:1]
            gt_W = gt_W[:1]
            self.workround = False

        wloss = self.Wloss(hat_W, gt_W)
        return wloss

    def layer(self, func, *data):
        try:
            return func(*data)
        except RuntimeError as e:
            print(e)
            raise OverflowError

    def plot(self, hat_W, gt_W, img, XYZ, batchepoch=(0, 0)):
        try:
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

                start = XYZ[img_index][:2]

                scaling_factor = 100
                end_hat_w = start + scaling_factor * np.array([hat_w[img_index][0], hat_w[img_index][1]])
                end_gt_w = start + scaling_factor * np.array([gt_w[img_index][0], gt_w[img_index][1]])

                # Create a set of points along the line for hat_w and gt_w
                num_points = 100
                points_hat_w = np.linspace(start, end_hat_w, num_points)
                points_gt_w = np.linspace(start, end_gt_w, num_points)

                # Plot hat_w line in green
                for i in range(num_points - 1):
                    ax.plot(points_hat_w[i:i + 2, 0], points_hat_w[i:i + 2, 1], '-', color='g', alpha=0.5)

                # Plot gt_w line in blue
                for i in range(num_points - 1):
                    ax.plot(points_gt_w[i:i + 2, 0], points_gt_w[i:i + 2, 1], '+', color='b', alpha=0.5)
                loss_value = self.Wloss(hat_W[img_index:img_index + 1], gt_W[img_index:img_index + 1], index=0)
                ax.set_title(f"{loss_value}")
                fig.savefig(f"xx_hn_d{batchepoch[1]}_{batchepoch[0]}_{img_index}.png")
                plt.close(fig)
        except Exception as e:
            print("wtf", e)
