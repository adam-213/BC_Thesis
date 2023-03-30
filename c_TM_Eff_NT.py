import random

import torch
import torch.nn as nn
import timm
import math


def _init_layer(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)


class PoseEstimationModel(nn.Module):
    def __init__(self, num_channels=7):
        super(PoseEstimationModel, self).__init__()

        self.Zloss = nn.MSELoss()
        self.Wloss = nn.MSELoss()

        self.backbone = timm.create_model("efficientnetv2_rw_s", pretrained=True, in_chans=num_channels)
        self.backbone.classifier = nn.Identity()

        self.Z_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1)
        )

        self.W_head = nn.Sequential(
            nn.Linear(self.backbone.num_features + 1, 2048),
            nn.GroupNorm(1, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.GroupNorm(1, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.GroupNorm(1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GroupNorm(1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )

        self.Z_head.apply(_init_layer)
        self.W_head.apply(_init_layer)

    def forward_s1(self, x):
        features = self.backbone(x)
        Z = self.Z_head(features)
        return Z

    def forward_s2(self, x):
        features = self.backbone(x)
        Z = self.Z_head(features)
        merged_features = torch.cat((features, Z), dim=1)
        W = self.W_head(merged_features)
        return W

    def loss_Z(self, hat_Z, gt_Z):
        return self.Zloss(hat_Z, gt_Z)

    def loss_W(self, What, Wgt):
        return self.Wloss(What, Wgt)

    def forward(self, x, stage):
        if stage == 1:
            return self.forward_s1(x)
        elif stage == 2:
            return self.forward_s2(x)


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
