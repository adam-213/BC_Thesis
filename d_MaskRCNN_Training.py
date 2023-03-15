from b_DataLoader_RCNN import createDataLoader
from c_MaskRCNN import MaskRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt


def targets2(targets, device):
    # transfer all targets  in the dictionary to device
    for target in targets:
        for key, val in target.items():
            target[key] = target[key].to(device)


def train(model, dataloader, optimizer, criterion):
    scaler = GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    losses_list = []
    for idx, (images, targets) in enumerate(dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        with autocast():
            loss_dict = model(images, targets)
            # loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        print(losses.item(), f"batch: {idx}/{len(dataloader)}, epoch: {epoch}")
        losses_list.append(losses.item())

    return losses_list


if __name__ == '__main__':
    from torch.cuda.amp import autocast, GradScaler

    dataset, dataloader = createDataLoader()
    mean, std = dataloader.dataset.mean[:7], dataloader.dataset.std[:7]
    model = MaskRCNN(7, 5, mean, std)
    # pytorch sum loss
    mask_loss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    scaler = GradScaler()

    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    lossess = []
    for epoch in range(5):
        lossess.extend(train(model, dataloader, optimizer, criterion))

        plt.plot(lossess)
        plt.show()

    # save model
    torch.save(model.state_dict(), "model_blender_5epoch_small32ds_nopretrain.pth")
