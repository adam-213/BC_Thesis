import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pathlib
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from b_DataLoader_RCNN import createDataLoader
from b_MaskRCNN import MaskRCNN
import numpy as np
import seaborn as sns
from copy import deepcopy


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler

    def uncuda(self, lossdict):
        for k, v in lossdict.items():
            lossdict[k] = v.cpu().detach().numpy().astype(np.float16)
        return lossdict

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
                loss_dict_list['total_loss'] = []

            loss_dict['total_loss'] = losses

            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            print(f"Loss: {losses.item()}, Batch: {idx}/{len(self.train_dataloader)}, Epoch: {epoch}")

            # self.scheduler.step_cosine_annealing(epoch)
            self.scheduler.step()
            total_loss_list.append(self.uncuda(loss_dict))

        return total_loss_list

    def validate(self, epoch):
        # self.model.eval() # this cant be used here because it will produce images not loss values
        total_loss_list = []

        with torch.no_grad():
            for idx, (images, targets) in enumerate(self.val_dataloader):
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                with autocast():
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                if idx == 0:
                    loss_dict_list = {k: [] for k in loss_dict.keys()}  # Initialize loss_dict_list here
                    loss_dict_list['total_loss'] = []

                loss_dict['total_loss'] = losses

                print(f"Validation Loss: {losses.item()}, Batch: {idx}/{len(self.val_dataloader)}, Epoch: {epoch}")
                for k, v in loss_dict.items():
                    loss_dict_list[k].append(v.item())
                loss_dict_list['total_loss'].append(losses.item())
                total_loss_list.append(self.uncuda(loss_dict))

        return total_loss_list

    def plot_losses(self, train_loss_dicts, val_loss_dicts, epoch=0):
        # Flatten the nested lists
        train_losses = [item for sublist in train_loss_dicts for item in sublist]
        val_losses = [item for sublist in val_loss_dicts for item in sublist]

        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        axes = axes.flatten()

        for idx, key in enumerate(train_losses[0].keys()):
            ax = axes[idx]

            train_loss_key = [loss_dict[key] for loss_dict in train_losses]
            val_loss_key = [loss_dict[key] for loss_dict in val_losses]

            train_loss_key = np.array(train_loss_key).astype(np.float64)
            val_loss_key = np.array(val_loss_key).astype(np.float64)

            # Remove or replace NaN values
            train_loss_key = np.nan_to_num(train_loss_key, nan=np.nanmean(train_loss_key))
            val_loss_key = np.nan_to_num(val_loss_key, nan=np.nanmean(val_loss_key))

            train_loss_len = len(train_loss_key)  # Calculate the offset for the validation loss curve
            val_loss_x = np.arange(train_loss_len, train_loss_len + len(val_loss_key))  # Add the offset to the x values

            # Add regression lines
            ax.plot(train_loss_key, label=f"Train {key}", color='lightblue')
            ax.plot(val_loss_x, val_loss_key, label=f"Validation {key}", color='lightcoral')

            sns.regplot(x=np.arange(len(train_loss_key)), y=train_loss_key, ax=ax, label=f"Train {key} RegLine",
                        color='blue',
                        scatter=False, order=4)
            sns.regplot(x=val_loss_x, y=val_loss_key, ax=ax, label=f"Validation {key} RegLine", color="orange",
                        scatter=False,
                        order=4)

            ax.set_title(f"Epoch {epoch} Losses for {key}")
            ax.set_xlabel("Batch")
            ax.set_ylabel("Loss")
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"maskrcnn_epoch{epoch}.png")
        plt.close()

    def train(self, num_epochs, checkpoint_path=None, load_checkpoint=None):
        if load_checkpoint:
            trainloss, valloss, epoch = self.load_checkpoint(load_checkpoint)
            epoch += 1 # Start from the next epoch because the checkpoint is saved after the epoch is finished
        else:
            trainloss, valloss = [], []
            epoch = 0
        write = False
        for epoch in range(epoch, num_epochs):
            train_losses = self.train_one_epoch(epoch)
            trainloss.append(train_losses)
            val_losses = self.validate(epoch)
            valloss.append(val_losses)

            trainloss = trainloss[-5:]
            valloss = valloss[-10:]

            self.plot_losses(trainloss, valloss, epoch)

            steploss = np.sum([loss["total_loss"] for loss in val_losses])

            #self.scheduler.step_reduce_on_plateau(torch.mean(torch.tensor(steploss)))

            # Save the checkpoint
            if checkpoint_path and epoch % 2 == 0 and write:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'train_losses': trainloss,
                    'val_losses': valloss,
                }, checkpoint_path.format(epoch))

            write = True  # prevent overriting the checkpoint on the first after loading / first epoch

        if checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'train_losses': trainloss,
                'val_losses': valloss,
            }, checkpoint_path.format(epoch))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # load scheduler state_dict
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        return train_losses, val_losses, epoch


from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


class IntegratedCosineAnnealingReduceOnPlateau:
    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1, factor=0.1, patience=2, verbose=True,
                 threshold=5e-4, cooldown=0, min_lr=0, eps=1e-8):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.cosine_annealing = CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch)
        self.reduce_on_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience,
                                                   verbose=verbose, threshold=threshold, cooldown=cooldown,
                                                   min_lr=min_lr, eps=eps)

    def step_cosine_annealing(self, epoch=None):
        self.cosine_annealing.step(epoch)

    def step_reduce_on_plateau(self, metrics):
        self.reduce_on_plateau.step(metrics)

    def get_lr(self):
        return self.cosine_annealing.get_lr()

    def reset_cosine_annealing(self, T_max=None, eta_min=None, last_epoch=None):
        T_max = T_max if T_max is not None else self.T_max
        eta_min = eta_min if eta_min is not None else min(self.get_lr())
        last_epoch = last_epoch if last_epoch is not None else self.last_epoch
        self.cosine_annealing = CosineAnnealingLR(self.optimizer, T_max, eta_min, last_epoch)

    def state_dict(self):
        return {
            'cosine_annealing': self.cosine_annealing.state_dict(),
            'reduce_on_plateau': self.reduce_on_plateau.state_dict(),
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'last_epoch': self.last_epoch
        }

    def load_state_dict(self, state_dict):
        self.cosine_annealing.load_state_dict(state_dict['cosine_annealing'])
        self.reduce_on_plateau.load_state_dict(state_dict['reduce_on_plateau'])
        self.T_max = state_dict['T_max']
        self.eta_min = state_dict['eta_min']
        self.last_epoch = state_dict['last_epoch']


def main():
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('known')
    channels = [0, 1, 2, 5, 9]

    train_dataloader, val_dataloader, stats = createDataLoader(coco_path, bs=3, num_workers=8,
                                                               channels=channels, split=0.9, shuffle=True)
    mean, std = stats
    mean, std = mean[channels], std[channels]
    cats = train_dataloader.dataset.dataset.coco.cats
    model = MaskRCNN(5, len(cats), mean, std)
    # model = MaskRCNN(5, 91, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler()
    lr = 3e-3
    num_epochs = 20

    #scheduler = IntegratedCosineAnnealingReduceOnPlateau(optimizer, T_max, eta_min, )

    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=0.002, eps=1e-8)

    # Use a cosine learning rate scheduler with a linear warm-up phase
    warmup_epochs = 3
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = warmup_epochs * len(train_dataloader)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps,
    #                                                 pct_start=warmup_steps / total_steps, anneal_strategy='cos')

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr,
                                                    total_steps=total_steps,
                                                    pct_start=warmup_steps / total_steps,
                                                    anneal_strategy='cos',
                                                    final_div_factor=600)

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler)
    save_path = "RCNN_Unscaled_{}.pth"
    trainer.train(num_epochs, save_path, load_checkpoint=None)


if __name__ == '__main__':
    main()
