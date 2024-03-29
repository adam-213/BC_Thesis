import time

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
import seaborn as sns
from d_Dataloader_TM_CNN_NT import createDataLoader
from d_Dataloader_Synth import createDataLoader as synthDataLoader
from d_TM_Eff_NT import PoseEstimationModel
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler
        self.plotbatch = 0

    def train_one_epoch_stage_2(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        total_loss_list = []
        for batch, (images_stacked_masked, z_vecs, XYZs) in enumerate(self.train_dataloader):
            if type(z_vecs) == type(None):
                continue

            images_stacked_masked = images_stacked_masked.to(self.device)
            z_vecs = z_vecs.to(self.device)
            # XYZs = XYZs.to(self.device)

            # Mixed precision training for speeeeed
            with autocast():
                weights = self.model(images_stacked_masked, XYZs)
                if weights is None:
                    print("None weights", images_stacked_masked.shape, XYZs.shape)
                    continue
                loss = self.model.loss_W(weights, z_vecs, False,
                                         images_stacked_masked, XYZs, [batch, epoch])

            # Scale the gradients
            try:
                self.scaler.scale(loss).backward()
            except Exception as e:
                print("Error in scaling gradients, skipping batch", e)
                continue
            # Update the optimizer with the combined gradients\
            # if batch %  == 0:
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            #     self.optimizer.zero_grad()
            #     self.scheduler.step()

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            # self.scheduler.step()

            # loss.backward()
            # self.optimizer.step()
            #
            # self.scheduler.step()

            total_loss_list.append(loss.item())

            print(f"Epoch {epoch} Batch {batch} Loss: {loss.item()}")
        return total_loss_list

    def val_one_epoch_stage_2(self, epoch):
        self.model.eval()
        total_val_loss_list = []
        self.plotbatch = self.plotbatch % epoch if epoch != 0 else 0

        with torch.no_grad():
            for batch, (images_stacked_masked, z_vecs, XYZs) in enumerate(
                    self.val_dataloader):  # TODO change to val_dataloader
                print(time.time())
                if type(z_vecs) == type(None):
                    continue
                with torch.no_grad():
                    self.optimizer.zero_grad()
                    images_stacked_masked = images_stacked_masked.to(self.device)
                    z_vecs = z_vecs.to(self.device)
                    # XYZs = XYZs.to(self.device)
                    plot = True if batch == self.plotbatch else False
                    # Mixed precision training for speeeeed
                    with autocast():
                        weights = self.model(images_stacked_masked, XYZs)
                        val_loss = self.model.loss_W(weights, z_vecs, plot, images_stacked_masked, XYZs, [batch, epoch])

                    total_val_loss_list.append(val_loss.item())

                    print(f"Validation Epoch {epoch} Batch {batch} Loss: {val_loss.item()}")

        self.plotbatch += 1
        return total_val_loss_list

    def train(self, num_epochs, stage=1, checkpoint_path=None, start_epoch=0):
        stage1_train_losses, stage1_val_losses = [], []
        stage2_train_losses, stage2_val_losses = [], []
        self.colors = sns.color_palette("husl", num_epochs)

        for epoch in range(start_epoch, num_epochs):
            s2_loss = self.train_one_epoch_stage_2(epoch)

            stage2_train_losses.append(s2_loss)

            s2_loss_val = self.val_one_epoch_stage_2(epoch)
            stage2_val_losses.append(s2_loss_val)
            # stage2_val_losses.append(s2_loss)
            # I had a crash on memory here so this is to prevent that
            # keep last 4 epochs
            stage2_train_losses = stage2_train_losses[-8:]
            # stage2_val_losses = stage2_val_losses[-4:]
            # self.scheduler.update_lr_on_plateau(np.mean(s2_loss_val))
            # self.scheduler.step_reduce_on_plateau(np.mean(s2_loss_val))
            t = time.time()
            self.plot_losses(stage2_train_losses, stage2_val_losses, stage=2, epoch=epoch, savepath=checkpoint_path)
            print("plotting took", time.time() - t)
            if epoch % 2 == 0:

                # Save the checkpoint
                if checkpoint_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scaler_state_dict': self.scaler.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    }, checkpoint_path.format(epoch, f"stage{stage}"))

    def plot_losses(self, train_losses, val_losses, stage=2, epoch=0, savepath=None):
        # Flatten the nested lists
        train_losses = [item for sublist in train_losses for item in sublist]
        val_losses = [item for sublist in val_losses for item in sublist]

        train_losses = np.array(train_losses).astype(np.float64)
        val_losses = np.array(val_losses).astype(np.float64)

        # Remove or replace NaN values
        train_losses = np.nan_to_num(train_losses, nan=np.nanmean(train_losses))
        val_losses = np.nan_to_num(val_losses, nan=np.nanmean(val_losses))

        fig, ax = plt.subplots(figsize=(10, 5))

        train_loss_len = len(train_losses)  # Calculate the offset for the validation loss curve
        val_loss_x = np.arange(train_loss_len, train_loss_len + len(val_losses))  # Add the offset to the x values

        # Add regression lines

        ax.plot(train_losses, label="Train")
        ax.plot(val_loss_x, val_losses, label="Validation",
                color=self.colors[epoch])  # Use the shifted x values for the validation loss curve
        ax.set_title(f"Stage {stage} Epoch {epoch} Losses")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        sns.regplot(x=np.arange(len(train_losses)), y=train_losses, ax=ax, label="Train RegLine", color='blue',
                    scatter=False, order=4)
        # choose a color for the validation loss curve based on the epoch number
        sns.regplot(x=val_loss_x, y=val_losses, ax=ax, label="Validation RegLine", color="orange", scatter=False,
                    order=4)
        ax.legend()
        if savepath:
            plt.savefig(savepath.format(epoch, f"stage{stage}") + "_losses_tuned.png")
        else:
            plt.savefig(f"efifv2rws{stage}_epoch{epoch}.png")
        plt.close()

    def load_checkpoint(self, path):
        """Use this when trying to resume training"""
        checkpoint = torch.load(path)
        # state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'W_head' not in k}
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        return epoch

    def load_checkpoint_tune(self, path):
        """Use this when changing the training parameters like for fine tuning"""
        checkpoint = torch.load(path)
        # state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'W_head' not in k}
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict, strict=False)


def train(base_path, coco_path, channels, gray):
    torch.random.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # train_dataloader, val_dataloader = createDataLoader(coco_path, batchsize=3, channels=channels, num_workers=10,
    #                                                     shuffle=False, gray=gray)

    train_dataloader, val_dataloader, _ = synthDataLoader(coco_path, bs=3, channels=channels, num_workers=10,
                                                           shuffle=True ,anoname="merged.json")

    model = PoseEstimationModel(len(channels) - 2 if gray else len(channels))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 40
    scaler = GradScaler()

    # Set the learning rate
    lr = 3e-3

    # Use the AdamW optimizer with weight decay
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    import torch_optimizer as optim

    # optimizer = optim.Lamb(model.parameters(), lr=lr, weight_decay=0.02, )

    optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=0.002, eps=1e-8)

    # Use a cosine learning rate scheduler with a linear warm-up phase
    warmup_epochs = 5
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
    from math import cos, pi

    # Initialize the trainer
    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler)
    # trainer.load_checkpoint("pose_estimation_model_VT175_5_stage2.pth")
    # trainer.scheduler.pct_start = 0.1

    save_path = "Unscaled__{}_{}.pth"

    trainer.train(num_epochs, checkpoint_path=save_path, stage=2)

    torch.save({
        'epoch': "full",
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, "full50.pth")


def tune(base_path, coco_path, channels, gray):
    # train_dataloader, val_dataloader = createDataLoader(coco_path, batchsize=3, channels=channels, num_workers=8,
    #                                                     shuffle=True, gray=gray)
    train_dataloader, val_dataloader, _ = synthDataLoader(coco_path, bs=2, channels=channels, num_workers=10,
                                                           shuffle=True)

    # model = PoseEstimationModel(len(channels) - 2 if gray else len(channels))
    model = PoseEstimationModel(len(channels) - 2 if gray else len(channels), tuning=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 20
    scaler = GradScaler()

    # Set the learning rate
    lr_backbone = 1e-5
    lr_head = 1e-5

    # Use the RAdam optimizer with weight decay
    optimizer = torch.optim.RAdam([
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': model.W_head.parameters(), 'lr': lr_head},
    ], weight_decay=0.0075)

    warmup_epochs = 2
    total_steps = (len(train_dataloader) - 7) * num_epochs
    warmup_steps = warmup_epochs * (len(train_dataloader) - 7)

    # Set maximum learning rates for the scheduler
    max_lr_backbone = 10 * lr_backbone
    max_lr_head = 15 * lr_head

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=[max_lr_backbone, max_lr_head],
                                                    total_steps=total_steps,
                                                    pct_start=warmup_steps / total_steps,
                                                    anneal_strategy='cos',
                                                    final_div_factor=1000)

    # Use a cosine learning rate scheduler with a linear warm-up phase
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Initialize the trainer
    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler)
    # trainer.load_checkpoint_tune("Unscaled_Tuned_4_stage2.pth")
    epoch = trainer.load_checkpoint("Unscaled_Tuned_6_stage2.pth")

    save_path = "Unscaled_Tuned_2_{}_{}.pth"

    trainer.train(num_epochs, checkpoint_path=save_path, stage=2, start_epoch=epoch)

    torch.save({
        'epoch': "full",
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, "full50.pth")


if __name__ == '__main__':
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('known')
    channels = [0, 1, 2, 5]
    gray = True
    # time.sleep(3000)
    train(base_path, coco_path, channels, gray)
    #tune(base_path, coco_path, channels, gray)
    # inference()
