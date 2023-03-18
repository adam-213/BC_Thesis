import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pathlib

import pathlib

from c_TM_CNN import TM_CNN
from b_DataLoader_RCNN import createDataLoader


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, scaler):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler

    def train(self, epochs, mini_batch_size):
        for epoch in range(epochs):
            running_loss = 0.0
            # batchsize must be 1
            for i, (images, masks, tms) in enumerate(self.train_loader):
                size = max(1, masks.size()[0] // mini_batch_size)
                for j in range(size):
                    # use torch to split the data into mini-batches
                    cut_size = min(mini_batch_size, masks.size()[0] - j * mini_batch_size)
                    masks_mini = masks[j * cut_size:(j + 1) * cut_size]
                    tms_mini = tms[j * cut_size:(j + 1) * cut_size]

                    # concat mask to the image on channel dimension for each image in mini-batch
                    # and stack them into the batch dimension

                    # create a temp list to store the batch
                    temp = []
                    for mask in masks_mini:
                        image = torch.cat((images.squeeze(), mask.unsqueeze(0)), dim=0)
                        temp.append(image)

                    # stack the batch
                    # should give me a tensor of size (mini_batch_size, imagechannles + 1, w,h)
                    images_mini = torch.stack(temp, dim=0)

                    images_mini = images_mini.to(self.device)
                    tms_mini = tms_mini.to(self.device)
                    assert images_mini.size()[0] == cut_size
                    assert tms_mini.size()[0] == cut_size

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    with autocast():
                        # forward + backward + optimize
                        outputs = self.model(images_mini)
                        loss = self.criterion(outputs, tms_mini)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()


if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [0, 1, 2, 3, 4, 5, 9]

    train_dataloader, val_dataloader, stats = createDataLoader(coco_path, 1, channels=channels,
                                                               model="tm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TM_CNN(in_channels=len(channels), attention_channels=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    trainer = Trainer(model, train_dataloader, val_dataloader, criterion, optimizer, device, scaler)
    trainer.train(epochs=50, mini_batch_size=16)
