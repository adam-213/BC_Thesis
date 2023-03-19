import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pathlib
import seaborn as sns
import numpy as np

from c_TM_CNN import TM_CNN
from b_DataLoader_RCNN import createDataLoader


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, scaler, scheduler):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler

    def train_one_epoch(self, epoch, batch_size):
        self.model.train()
        total_loss_list = []
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

                    # print(image_stacked.shape, minimasks.shape, minitms.shape)

                    # stack minimasks to the image as a channel
                    image_masks = torch.cat((image_stacked, minimasks.unsqueeze(1)), dim=1)

                    image_masks = image_masks.to(self.device)
                    # minimasks = minimasks.to(self.device)
                    minitms = minitms.to(self.device)

                    with autocast():
                        outputs = self.model(image_masks)
                        loss = self.model.loss(outputs, minitms)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    running_loss += loss.item()
                    total_loss_list.append(loss.item())

                    print(f"Loss: {loss.item()}, Batch: {i}/{len(self.train_loader)}, Epoch: {epoch}")
            running_loss = 0.0
            self.plot_losses(total_loss_list, epoch, i, len(self.train_loader))

        return total_loss_list

    def validate(self):
        self.model.eval()
        total_loss_list = []
        running_loss = 0.0
        batch_size = 4

        with torch.no_grad():
            for i, (images, masks, tms) in enumerate(self.val_loader):
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

                        # transform minitms to tranlation and rotation quaternion




                        image_stacked = torch.stack([image] * minimasks.shape[0])

                        self.optimizer.zero_grad()

                        # print(image_stacked.shape, minimasks.shape, minitms.shape)

                        # stack minimasks to the image as a channel
                        image_masks = torch.cat((image_stacked, minimasks.unsqueeze(1)), dim=1)

                        image_masks = image_masks.to(self.device)
                        # minimasks = minimasks.to(self.device)
                        minitms = minitms.to(self.device)

                        outputs = self.model(image_masks)
                        loss = self.model.loss(outputs, minitms)

                        running_loss += loss.item()
                        print(f"VAL Loss: {loss.item()}, Batch: {i}/{len(self.val_loader)}")

            total_loss_list.append(running_loss / (slices * len(self.val_loader)))

        return total_loss_list

    def plot_losses(self, loss_list, epoch, batch_idx, total_batches, interval=10):
        if batch_idx % interval == 0 and len(loss_list) > 250:
            loss_list = loss_list[300:]
            clear_output(wait=True)
            plt.figure(figsize=(10, 5))
            plt.plot(loss_list, label='Total Loss')

            # Get x and y data for the regplot
            x_data = list(range(len(loss_list)))
            y_data = loss_list

            # Add a regplot (regression line) to the plot
            sns.regplot(x=x_data, y=y_data, scatter=False, line_kws={'color': 'red', 'label': 'Regression Line'})

            plt.xlabel('Microbatches')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Epoch {epoch}, Batch {batch_idx}/{total_batches}')
            plt.grid()
            plt.show()

    def train(self, epochs, batch_size, checkpoint_path=None):
        for epoch in range(epochs):
            train_losses = self.train_one_epoch(epoch, batch_size)
            val_losses = self.validate()

            self.plot_losses(train_losses, epoch, len(self.train_loader), len(self.train_loader))

            # Save the checkpoint
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, checkpoint_path.format(epoch))

            # schedule the learning rate
            self.scheduler.step(np.mean(val_losses))

    def load_checkpoint(self, model, optimizer, scaler, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        return epoch, train_losses, val_losses


if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [0, 1, 2, 3, 4, 5, 9]
    train_dataloader, val_dataloader, stats = createDataLoader(coco_path, 16, channels=channels, model="tm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TM_CNN(in_channels=len(channels) + 1, d_model=512, nhead=8, num_layers=1).to(device)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    scaler = GradScaler()

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler)
    trainer.train(epochs=50, batch_size=4, checkpoint_path="TM_CNN_{}.pth")

# Be aware that matrices are predicted in the other order not the one that is expexted  by the generator
# not the reshape(4,4,order='F') but reshape(4,4,order='C')
