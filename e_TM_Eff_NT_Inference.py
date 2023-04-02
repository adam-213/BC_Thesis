import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pathlib
from torch.utils.checkpoint import checkpoint
import random
import numpy as np

from b_Dataloader_TM_CNN_NT import createDataLoaderM
from c_TM_Eff_NT import PoseEstimationModel



model.cuda()
model.eval()

for i, (images, masks, rot, move, names) in enumerate(train_dataloader):
    for idx, image in enumerate(images):
        for j, (masks, rot, move, names) in enumerate(zip(masks, rot, move, names)):
            image = image.cuda().float()
            masks = masks.cuda().float()
            rot = rot.cuda().float()
            move = move.cuda().float()

            minimasks.squeeze_(0)
            minirot.squeeze_(0)
            minimove.squeeze_(0)

            # stack the image to the same size as the masks
            if len(minimasks.shape) != 3:
                # Workaround in case the microbatch size is 1
                minimasks = minimasks.unsqueeze(0)
                minirot = minirot.unsqueeze(0)
                minimove = minimove.unsqueeze(0)

            # copy the image to match the microbatch size
            image_stacked = torch.stack([image] * minimasks.shape[0])
            # stack minimasks to the image as a channel
            # image_masks_stacked = torch.cat((image_stacked, minimasks.unsqueeze(1)), dim=1)
            images_rgb, images_aux = image_stacked[:, :3], image_stacked[:, 3:]
            images_rgb_masked = torch.cat((images_rgb, minimasks.unsqueeze(1)), dim=1)
            images_aux_masked = images_aux * minimasks.unsqueeze(1)
            del image_stacked, images_rgb, images_aux

            image_masks_stacked = torch.cat((images_rgb_masked, images_aux_masked), dim=1)
            del images_rgb_masked, images_aux_masked
            # move to device
            image_masks_stacked = image_masks_stacked.to(self.device)
            minirot = minirot.to(self.device)
            minimove = minimove.to(self.device)
            # mininames = mininames.to(self.device)

            # turn on grad for input
            image_masks_stacked.requires_grad_(True)

            self.optimizer.zero_grad()

            # with autocast():
            movehat, rothat = checkpoint(self.model, image_masks_stacked)
