from b_DataLoader import createDataLoader
from c_FPN50 import gen_fcn_fpn50

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, mask_logits, targets):
        # Compute the binary cross-entropy loss
        loss = nn.functional.binary_cross_entropy_with_logits(mask_logits, targets)
        return loss


if __name__ == '__main__':
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    dataset, dataloader = createDataLoader()
    model = gen_fcn_fpn50(3, 3)
    mask_loss = MaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.cuda()
    mask_loss.cuda()

    for i, batch in enumerate(tqdm(dataloader)):
        with autocast():
            images, targets = batch[0].cuda(), batch[1]
            for target in targets:
                for key, val in target.items():
                    target[key] = target[key].cuda()
            del batch

            # forward pass
            outputs = model(images,targets)
            # loss
        #     loss = mask_loss.forward(outputs, targets)
        #
        # # backward pass
        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        #
        # # update weights
        # scaler.step(optimizer)
        # scaler.update()
