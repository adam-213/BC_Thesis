import time

from b_DataLoader_RCNN import createDataLoader
from c_MaskRCNN import MaskRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, mask_logits, targets):
        # Compute the binary cross-entropy loss
        # loss = nn.functional.binary_cross_entropy_with_logits(mask_logits, targets)
        x = torch.tensor(0.00)
        x.requires_grad_(True)
        return x.cuda()


def targets2(targets, device):
    for target in targets:
        for key, val in target.items():
            target[key] = target[key].to(device)


if __name__ == '__main__':
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    dataset, dataloader = createDataLoader()
    model = MaskRCNN(3, 3)
    mask_loss = MaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.cuda()
    mask_loss.cuda()
    for i, batch in enumerate(tqdm(dataloader)):
        with autocast():
            images, targets = batch[0], batch[1]
            images = images.cuda()
            targets2(targets, 'cuda')
            del batch
            # torch.cuda.reset_max_memory_allocated()
            # print("to cuda",torch.cuda.max_memory_allocated())
            # time.sleep(10)

            # forward pass
            outputs = checkpoint(model, images, targets)

            # compute loss
            loss = mask_loss(outputs, targets)
            # del outputs
            # del targets
            # del images

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
