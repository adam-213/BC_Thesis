import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
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

    def train(self, epochs, batch_size):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (images, masks, tms) in enumerate(self.train_loader):
                for idx, image in enumerate(images):
                    slices = max(1, masks[idx].shape[0] // batch_size)
                    for slice in range(slices):
                        minimasks = masks[idx][slice:slice + batch_size]
                        minitms = tms[idx][slice:slice + batch_size]

                        minimasks.squeeze_(0)
                        minitms.squeeze_(0)
                        # stack the image to the same size as the masks
                        if len(minimasks.shape) != 3:
                            minimasks = minimasks.unsqueeze(0)
                            minitms = minitms.unsqueeze(0)

                        image_stacked = torch.stack([image] * minimasks.shape[0])

                        self.optimizer.zero_grad()

                        #print(image_stacked.shape, minimasks.shape, minitms.shape)

                        image_stacked = image_stacked.to(self.device)
                        minimasks = minimasks.to(self.device)
                        minitms = minitms.to(self.device)



                        with autocast():
                            outputs = self.model(image_stacked, minimasks)
                            loss = self.model.loss(outputs, minitms)

                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                        running_loss += loss.item()

                print(f"Loss: {running_loss / slices}, Batch: {i}/{len(self.train_loader)}, Epoch: {epoch}")
                running_loss = 0.0


if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [0, 1, 2, 3, 4, 5, 9]
    train_dataloader, val_dataloader, stats = createDataLoader(coco_path, 4, channels=channels,
                                                               model="tm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TM_CNN(in_channels=len(channels), attention_channels=64).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler)
    trainer.train(epochs=50, batch_size=4)
