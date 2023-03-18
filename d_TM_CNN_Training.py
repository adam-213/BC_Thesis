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
    def __init__(self, model, train_loader, val_loader, optimizer, device, scaler):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler

    def train(self, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (images, masks, tms) in enumerate(self.train_loader):
                # split the batch into mini-batches
                images = images.to(self.device)
                masks = masks.to(self.device)
                tms = tms.to(self.device)

                for minimini_batch in range(0, images.shape[0], 2):
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    with autocast():
                        # forward + backward + optimize
                        outputs = self.model(images[minimini_batch:minimini_batch + 2],
                                             masks[minimini_batch:minimini_batch + 2])
                        print("stop")
                        loss = self.model.loss(outputs, tms[minimini_batch:minimini_batch + 2])

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # print statistics
                    running_loss += loss.item()
                    if i % 10 == 9:
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 10))
                        running_loss = 0.0

                # # zero the parameter gradients
                # self.optimizer.zero_grad()
                #
                # with autocast():
                #     # forward + backward + optimize
                #     outputs = self.model(images, masks)
                #     loss = self.criterion(outputs, tms)
                #
                # self.scaler.scale(loss).backward()
                # self.scaler.step(self.optimizer)
                # self.scaler.update()


if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [0, 1, 2, 3, 4, 5, 9]
    instances = 1
    train_dataloader, val_dataloader, stats = createDataLoader(coco_path, 1, channels=channels,
                                                               model="tm", instances=instances)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TM_CNN(in_channels=len(channels), attention_channels=32, instances=instances).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler)
    trainer.train(epochs=50)
