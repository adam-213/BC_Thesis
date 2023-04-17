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
    def __init__(self, alpha=0.5, lambda_magnitude=0.1, lambda_range=1.0, use_geodesic_loss=True, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.lambda_magnitude = lambda_magnitude
        self.lambda_range = lambda_range
        self.use_geodesic_loss = use_geodesic_loss
        self.beta = beta

    def geodesic_loss(self, u, v):
        dot_product = torch.sum(u * v, dim=1)
        norm_u = torch.norm(u, p=2, dim=1)
        norm_v = torch.norm(v, p=2, dim=1)
        angle_cosine = dot_product / (norm_u * norm_v)
        angle_cosine = torch.clamp(angle_cosine, -1.0, 1.0)
        return torch.acos(angle_cosine)

    def squared_euclidean_loss(self, u, v):
        return torch.sum((u - v) ** 2, dim=1)

    def MSE(self, u, v):
        return torch.sum((u - v) ** 2, dim=1)

    def magnitude_regularization(self, u):
        return torch.abs(torch.norm(u, p=2, dim=1) - 1)

    def range_regularization(self, u):
        return torch.sum(torch.relu(torch.abs(u) - 1) ** 2)

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

        mse = self.MSE(pred_norm, target)

        # Weighted combination of the main loss, magnitude regularization, and range regularization
        combined_loss = self.alpha * main_loss \
                        + self.lambda_magnitude * mag_reg \
                        + self.lambda_range * range_reg \
                        + self.beta * mse

        if index is not None:
            # Return the loss for a single instance for plotting
            return combined_loss[index]
        else:
            return torch.mean(combined_loss)


class PoseEstimationModel(nn.Module):
    def __init__(self, num_channels=7):
        super(PoseEstimationModel, self).__init__()
        self.workround = False

        self.Wloss = CustomLoss(use_geodesic_loss=True,
                                alpha=1.0,
                                lambda_magnitude=0.0,
                                lambda_range=0.0,
                                beta=0)
        # self.Wloss = torch.nn.MSELoss()
        self.input_norm = nn.BatchNorm2d(num_channels)
        # self.rotation_equivariant_layer = RotationEquivariantLayer(num_channels, mid_channels, num_rotations)
        self.backbone = timm.create_model("deit3_medium_patch16_224", pretrained=False,
                                          in_chans=num_channels, num_classes=0)

        self.backbone.head = nn.Identity()

        self.W_head = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, 384),
            nn.BatchNorm1d(384),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(384, 160),
            nn.Hardswish(),
            nn.Dropout(0.1),
            nn.Linear(160, 3),
            nn.Tanh()  # normalize to [-1, 1] as the output is a direction vector aka a unit vector
        )

        self.W_head.apply(_init_layer)
        self.backbone.apply(_init_layer)

    def forward(self, x, XYZ):
        # test if every shape is divisible by 16
        # if not, pad it
        # print(x.shape)

        if x.shape[0] <= 1:
            x = torch.cat((x, x), dim=0)
            XYZ = torch.cat((XYZ, XYZ), dim=0)
            self.workround = True
        try:
            x = self.layer(self.input_norm, x)
            backbonex = self.layer(self.backbone, x)
            # x = torch.cat((backbonex, XYZ), dim=1)
            x = self.layer(self.W_head, backbonex)
            return x
        except OverflowError as e:
            print(e)
            return None

    def loss_W(self, hat_W, gt_W, plot, *plotargs):
        # normalisation, but im not sure if its helping or hurting
        # magnitude = torch.sqrt(torch.sum(hat_W ** 2, dim=1)).view(-1, 1)
        # hat_W = hat_W / magnitude
        if random.random() < 0.001: self.plot(hat_W, gt_W, *plotargs)
        if plot: self.plot(hat_W, gt_W, *plotargs)

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

    from mpl_toolkits.mplot3d import Axes3D

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

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                ax1.imshow(cur_img)

                start = np.array([cur_img.shape[1] // 2, cur_img.shape[0] // 2])
                scaling_factor = 100
                end_hat_w = start + scaling_factor * np.array([hat_w[img_index][0], hat_w[img_index][1]])
                end_gt_w = start + scaling_factor * np.array([gt_w[img_index][0], gt_w[img_index][1]])

                # Create a set of points along the line for hat_w and gt_w
                num_points = 100
                points_hat_w = np.linspace(start, end_hat_w, num_points)
                points_gt_w = np.linspace(start, end_gt_w, num_points)

                # Plot the lines in the original image (ax1)
                for i in range(num_points - 1):
                    ax1.plot(points_hat_w[i:i + 2, 0], points_hat_w[i:i + 2, 1], '+', color='g', alpha=0.5)
                    ax1.plot(points_gt_w[i:i + 2, 0], points_gt_w[i:i + 2, 1], '+', color='b', alpha=0.5)
                # 3D subplot with arcs
                ax2 = fig.add_subplot(122, projection='3d')
                # Draw octant separation planes with different colors and 0.2 alpha
                xx, yy = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2))
                zz = np.zeros_like(xx)

                ax2.plot_surface(xx, yy, zz, alpha=0.1, color='r')  # X-Y plane (red)
                ax2.plot_surface(xx, zz, yy, alpha=0.1, color='g')  # X-Z plane (green)
                ax2.plot_surface(zz, yy, xx, alpha=0.1, color='b')  # Y-Z plane (blue)
                # Plot hat_w and gt_w vectors
                ax2.quiver(0, 0, 0, hat_w[img_index][0], hat_w[img_index][1], hat_w[img_index][2], color='g', alpha=0.5)
                ax2.quiver(0, 0, 0, gt_w[img_index][0], gt_w[img_index][1], gt_w[img_index][2], color='b', alpha=0.5)

                # Calculate the arc using Spherical Linear Interpolation (SLERP)
                num_points = 100
                arc_points = np.array(
                    [slerp(hat_w[img_index], gt_w[img_index], t) for t in np.linspace(0, 1, num_points)])
                # Plot the arc
                ax2.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color='r', alpha=0.5)

                # Set the limits and aspect ratio
                ax2.set_xlim(-1, 1)
                ax2.set_ylim(-1, 1)
                ax2.set_zlim(-1, 1)
                ax2.set_box_aspect((1, 1, 1))

                loss_value = self.Wloss(hat_W[img_index:img_index + 1], gt_W[img_index:img_index + 1], index=0)
                ax1.set_title(f"{loss_value}")

                fig.savefig(f"xx_hn_d{batchepoch[1]}_{batchepoch[0]}_{img_index}.png")
                plt.close(fig)

        except Exception as e:
            print("wtf", e)


def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1
