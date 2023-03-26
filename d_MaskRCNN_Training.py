import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pathlib
from torch.optim.lr_scheduler import CosineAnnealingLR
from b_DataLoader_RCNN import createDataLoader
from c_MaskRCNN import MaskRCNN


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss_list = []

        for idx, (images, targets) in enumerate(self.train_dataloader):
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            with autocast():
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if idx == 0:
                loss_dict_list = {k: [] for k in loss_dict.keys()}  # Initialize loss_dict_list here

            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            print(f"Loss: {losses.item()}, Batch: {idx}/{len(self.train_dataloader)}, Epoch: {epoch}")
            total_loss_list.append(losses.item())
            for key, value in loss_dict.items():
                loss_dict_list[key].append(value.item())

            self.plot_losses(total_loss_list, loss_dict_list, epoch, idx, len(self.train_dataloader))

            if self.scheduler and idx % 200 and idx != 0:
                self.scheduler.step()

        return total_loss_list

    def validate(self, epoch):
        # self.model.eval()
        total_loss_list = []

        with torch.no_grad():
            for idx, (images, targets) in enumerate(self.val_dataloader):
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                if idx == 0:
                    loss_dict_list = {k: [] for k in loss_dict.keys()}  # Initialize loss_dict_list here

                print(f"Validation Loss: {losses.item()}, Batch: {idx}/{len(self.val_dataloader)}, Epoch: {epoch}")
                total_loss_list.append(losses.item())
                for key, value in loss_dict.items():
                    loss_dict_list[key].append(value.item())

                self.plot_losses(total_loss_list, loss_dict_list, epoch, idx, len(self.val_dataloader), validation=True)

        return total_loss_list, loss_dict_list

    def plot_losses(self, loss_list, loss_dict, epoch, batch_idx, total_batches, interval=50, validation=False):
        if batch_idx % interval == 0:
            clear_output(wait=True)
            plt.figure(figsize=(10, 5))
            plt.plot(loss_list, label='Total Loss' + (' (Validation)' if validation else ''))

            for key, value in loss_dict.items():
                plt.plot(value, label=f'{key} Loss' + (' (Validation)' if validation else ''))

            plt.xlabel('Batches')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Epoch {epoch}, Batch {batch_idx}/{total_batches}' + (' (Validation)' if validation else ''))
            plt.grid()
            plt.savefig(f'losses{batch_idx}_{epoch}.png')

    def train(self, num_epochs, checkpoint_path=None):
        for epoch in range(num_epochs):
            train_losses = self.train_one_epoch(epoch)
            val_losses, val_loss_dict = self.validate(epoch)

            self.plot_losses(train_losses, val_loss_dict, epoch, len(self.train_dataloader), len(self.train_dataloader))

            # Save the checkpoint
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_loss_dict': val_loss_dict
                }, checkpoint_path.format(epoch))

    def load_checkpoint(self, checkpoint_path, model):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scaler.load_state_dict(checkpoint['scaler_state_dict'])
        # epoch = checkpoint['epoch']
        # train_losses = checkpoint['train_losses']
        # val_losses = checkpoint['val_losses']
        # val_loss_dict = checkpoint['val_loss_dict']
        return model

def main():
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [0, 1, 2, 5, 9]

    train_dataloader, val_dataloader, stats = createDataLoader(coco_path, 4, channels=channels)
    mean, std = stats
    mean, std = mean[channels], std[channels]

    model = MaskRCNN(5, 5, mean, std)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler()
    batches_per_epoch = len(train_dataloader)
    batches_per_cycle = 200
    num_epochs = 100
    T_max = batches_per_epoch * num_epochs // batches_per_cycle
    eta_min = 1e-6
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler)
    # just so it runs basically forever, you can stop it whenever you want - checkpoints are saved every epoch
    save_path = "RCNN_TM_{}.pth"
    trainer.train(num_epochs * 2, save_path)


if __name__ == '__main__':
    main()
