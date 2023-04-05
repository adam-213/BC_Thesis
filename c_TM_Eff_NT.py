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

        self.Zloss = nn.MSELoss()
        self.Wloss = nn.MSELoss()

        self.backbone = timm.create_model("mobilevitv2_100", pretrained=False, in_chans=num_channels)
        self.backbone.classifier = nn.Identity()

        self.W_head = nn.Sequential(
            nn.Linear(1000 + 3, 4096+1024),
            nn.BatchNorm1d(4096+1024),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.5),
            nn.Linear(4096+1024, 2048+512),
            nn.BatchNorm1d(2048+512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(2048+512, 1024+256),
            nn.BatchNorm1d(1024+256),
            nn.LeakyReLU(0.15),
            nn.Dropout(0.5),
            nn.Linear(1024+256, 512+128),
            nn.LeakyReLU(0.1),
            nn.Linear(512+128, 3)
        )

        self.W_head.apply(_init_layer)
        self.backbone.apply(_init_layer)

    def forward_s1(self, x):
        features = self.backbone(x)
        Z = self.Z_head(features)
        return Z

    def forward_s2(self, x, XYZ):
        if x.shape[0] == 1:
            # workround for batch size 1, as batchnorm does not work with batch size 1
            x = torch.cat([x, x], dim=0)
            XYZ = torch.cat([XYZ, XYZ], dim=0)
            self.workround = True
        features = self.backbone(x)
        stacked = torch.cat((features, XYZ), dim=1)
        W = self.W_head(stacked)
        return W

    def loss_Z(self, hat_Z, gt_Z):
        return self.Zloss(hat_Z, gt_Z)

    def loss_W(self, hat_W, gt_W, img=None, XYZ=None,batchepoch=(0,0)):
        if self.workround:
            # hat_W = hat_W[0]
            # gt_W = gt_W[0]
            self.workround = False

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

                Img = Img[:3, :, :]

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
                        ax.plot(points_hat_w[i:i + 2, 0], points_hat_w[i:i + 2, 1], '-', color='g', alpha=0.8)

                    # Plot gt_w line in blue
                    for i in range(num_points - 1):
                        ax.plot(points_gt_w[i:i + 2, 0], points_gt_w[i:i + 2, 1], '.', color='b', alpha=0.8)

                    fig.savefig(f"VT_test_{batchepoch[0]}_{batchepoch[1]}_{img_index}.png")
                    plt.close(fig)
        except:
            pass

        wloss = self.Wloss(hat_W, gt_W)
        return wloss




    def forward(self, x, stage):
        if stage == 1:
            return self.forward_s1(x)
        elif stage == 2:
            return self.forward_s2(x)

    def normalize_rows(self, tensor):
        return tensor / torch.norm(tensor, dim=1, keepdim=True)


class RotationLoss():

    def __init__(self):
        self.step = 10

    def __call__(self, hat_rot, gt_rot, labelname):
        # this is a hack, don't do this, in current implementation all parts in the batch must be of the same type
        labelname = labelname[0]
        if labelname == "BACKROUND_UNDEFINED":
            return torch.zeros(1)
        elif "bin" in labelname.lower():
            return torch.zeros(1)
        elif "part_" in labelname.lower():
            return self.part_handler(labelname, hat_rot, gt_rot)

    def part_handler(self, labelname, hat_rot, gt_rot):
        if labelname == "part_thruster":
            return self.thruster(hat_rot, gt_rot)
        elif labelname == "part_cogwheel":
            return self.cogwheel(hat_rot, gt_rot)
        else:
            return self.any_other_part(hat_rot, gt_rot)

    def thruster(self, hat_rot, gt_rot):
        # thruster has a symmetry through the long axis
        # can be simplified to a cylinder
        rot_loss = self.geodesic_loss(hat_rot, gt_rot)
        best_rot_loss = torch.tensor(float('inf')).to(rot_loss.device)
        best_angle = 0

        # First pass: find the best 5-degree step
        for i in range(0, 360, self.step):
            gt_rot_rotated = gt_rot.clone()
            gt_rot_rotated[:, 2] += i * (2 * torch.pi / 360)  # Add i degrees to the z-axis angle
            gt_rot_rotated[:, 2] %= 2 * torch.pi  # Wrap the z-axis rotation angle
            rot_loss_rotated = self.geodesic_loss(hat_rot, gt_rot_rotated)
            if torch.lt(rot_loss_rotated, best_rot_loss).all():
                best_rot_loss = rot_loss_rotated
                best_angle = i

        # Second pass: narrow it down with 1-degree steps around the best 5-degree step
        min_rot_loss = best_rot_loss
        for i in range(best_angle - self.step, best_angle + self.step + 1):
            gt_rot_rotated = gt_rot.clone()
            gt_rot_rotated[:, 2] += i * (2 * torch.pi / 360)  # Add i degrees to the z-axis angle
            gt_rot_rotated[:, 2] %= 2 * torch.pi  # Wrap the z-axis rotation angle
            rot_loss_rotated = self.geodesic_loss(hat_rot, gt_rot_rotated)
            min_rot_loss = torch.min(min_rot_loss, rot_loss_rotated)

        return min_rot_loss

    def cogwheel(self, hat_rot, gt_rot):
        # cogwheel has a "xy mirror" symmetry and a rotational symmetry around the z-axis
        rot_loss = self.geodesic_loss(hat_rot, gt_rot)
        best_rot_loss = torch.tensor(float('inf')).to(rot_loss.device)
        best_angle = 0

        # First pass: find the best 5-degree step
        for i in range(0, 360, self.step):
            gt_rot_rotated = gt_rot.clone()
            gt_rot_rotated[:, 2] += i * (2 * torch.pi / 360)  # Add i degrees to the z-axis angle
            gt_rot_rotated[:, 2] %= 2 * torch.pi  # Wrap the z-axis rotation angle
            rot_loss_rotated = self.geodesic_loss(hat_rot, gt_rot_rotated)
            if torch.lt(rot_loss_rotated, best_rot_loss).all():
                best_rot_loss = rot_loss_rotated
                best_angle = i

        # Second pass: narrow it down with 1-degree steps around the best 5-degree step
        min_rot_loss = best_rot_loss
        for i in range(best_angle - self.step, best_angle + self.step + 1):
            gt_rot_rotated = gt_rot.clone()
            gt_rot_rotated[:, 2] += i * (2 * torch.pi / 360)  # Add i degrees to the z-axis angle
            gt_rot_rotated[:, 2] %= 2 * torch.pi  # Wrap the z-axis rotation angle
            rot_loss_rotated = self.geodesic_loss(hat_rot, gt_rot_rotated)
            min_rot_loss = torch.min(min_rot_loss, rot_loss_rotated)

        # Mirror symmetry
        gt_rot_mirrored = gt_rot.clone()
        gt_rot_mirrored[:, 0] *= -1  # mirror the x-axis
        gt_rot_mirrored[:, 1] *= -1  # mirror the y-axis to get the correct mirrored rotation
        rot_loss_mirrored = self.geodesic_loss(hat_rot, gt_rot_mirrored)

        # Combine the minimum rotational loss and mirror loss
        final_min_loss = torch.min(min_rot_loss, rot_loss_mirrored)

        return final_min_loss

    def any_other_part(self, hat_rot, gt_rot):
        # Not implemented, shouldn't be needed, but if it is, it is probably a bug
        raise NotImplementedError

    def geodesic_loss(self, hat_rot, gt_rot):
        # Normalize the rotation axes
        hat_rot_normalized = hat_rot / hat_rot.norm(dim=1, keepdim=True)
        gt_rot_normalized = gt_rot / gt_rot.norm(dim=1, keepdim=True)

        # Calculate the angle between the rotation axes
        cos_angle_diff = (hat_rot_normalized * gt_rot_normalized).sum(dim=1)
        angle_diff = torch.acos(torch.clamp(cos_angle_diff, -1, 1))

        return angle_diff
