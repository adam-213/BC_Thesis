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
from b_Dataloader_TM_CNN_NT import createDataLoader
from c_TM_Eff_NT import PoseEstimationModel


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler

    def train_one_epoch_stage_1(self, epoch):
        self.model.train()
        total_loss_list = []

        for batch, (images_stacked_masked, z, z_vecs, names) in enumerate(self.train_dataloader):
            if type(z) == type(None):
                continue
            self.optimizer.zero_grad()

            images_stacked_masked = images_stacked_masked.to(self.device)
            z = z.to(self.device)

            # Checkpointing for memory efficiency
            # Z = checkpoint(self.model.forward_s1, image_masks_stacked)
            # loss = self.model.loss_Z(Z, zs)
            # loss.bckward()

            # self.optimizer.step()
            # self.optimizer.zero_grad()

            # Mixed precision training for speeeeed
            with autocast():
                Z = self.model.forward_s1(images_stacked_masked)
                loss = self.model.loss_Z(Z, z)

            # Scale the gradients
            self.scaler.scale(loss).backward()
            # Update the optimizer with the combined gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_list.append(loss.item())

            print(f"Epoch {epoch} Batch {batch} Loss: {loss.item()}")

        return total_loss_list

    def val_one_epoch_stage_1(self, epoch):
        self.model.eval()
        total_val_loss_list = []

        with torch.no_grad():
            for batch, (images_stacked_masked, z, z_vecs, names) in enumerate(self.val_dataloader):
                if type(z) == type(None):
                    continue
                images_stacked_masked = images_stacked_masked.to(self.device)
                z = z.to(self.device)

                # Mixed precision validation
                with autocast():
                    Z = self.model.forward_s1(images_stacked_masked)
                    val_loss = self.model.loss_Z(Z, z)

                total_val_loss_list.append(val_loss.item())

                print(f"Validation Epoch {epoch} Batch {batch} Loss: {val_loss.item()}")

        return total_val_loss_list

    def train_one_epoch_stage_2(self, epoch):
        self.model.train()
        total_loss_list = []
        for batch, (images_stacked_masked, z, z_vecs, names, XYZs) in enumerate(self.train_dataloader):
            if type(z) == type(None):
                continue
            self.optimizer.zero_grad()
            z = z.to(self.device)
            images_stacked_masked = images_stacked_masked.to(self.device)
            z_vecs = z_vecs.to(self.device)
            XYZs = XYZs.to(self.device)

            # Mixed precision training for speeeeed

            weights = self.model.forward_s2(images_stacked_masked, XYZs)
            loss = self.model.loss_W(weights, z_vecs, images_stacked_masked, XYZs, [batch, epoch])

            # Scale the gradients
            # self.scaler.scale(loss).backward()
            # # Update the optimizer with the combined gradients
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            # self.scheduler.step()

            loss.backward()
            self.optimizer.step()

            self.scheduler.step()

            total_loss_list.append(loss.item())

            print(f"Epoch {epoch} Batch {batch} Loss: {loss.item()}")

        return total_loss_list

    def val_one_epoch_stage_2(self, epoch):
        self.model.eval()
        total_val_loss_list = []

        with torch.no_grad():
            for batch, (images_stacked_masked, z, z_vecs, names, XYZs) in enumerate(self.val_dataloader):
                if type(z) == type(None):
                    continue
                self.optimizer.zero_grad()
                z = z.to(self.device)
                images_stacked_masked = images_stacked_masked.to(self.device)
                z_vecs = z_vecs.to(self.device)
                XYZs = XYZs.to(self.device)

                # Mixed precision training for speeeeed
                with autocast():
                    weights = self.model.forward_s2(images_stacked_masked, XYZs)
                    val_loss = self.model.loss_W(weights, z_vecs, images_stacked_masked, XYZs, [batch, epoch])

                total_val_loss_list.append(val_loss.item())

                print(f"Validation Epoch {epoch} Batch {batch} Loss: {val_loss.item()}")

        return total_val_loss_list

    def train(self, num_epochs, stage=1, checkpoint_path=None):
        stage1_train_losses, stage1_val_losses = [], []
        stage2_train_losses, stage2_val_losses = [], []
        for epoch in range(num_epochs):
            if stage == 1:
                s1_loss = self.train_one_epoch_stage_1(epoch)
                stage1_train_losses.append(s1_loss)

                s1_loss_val = self.val_one_epoch_stage_1(epoch)
                stage1_val_losses.append(s1_loss_val)

                self.plot_losses(stage1_train_losses, stage1_val_losses, stage=1, epoch=epoch)

                self.scheduler.step(np.mean(stage1_val_losses))

            elif stage == 2:
                s2_loss = self.train_one_epoch_stage_2(epoch)
                stage2_train_losses.append(s2_loss)

                s2_loss_val = self.val_one_epoch_stage_2(epoch)
                stage2_val_losses.append(s2_loss_val)
                # stage2_val_losses.append(s2_loss)
                # I had a crash on memory here so this is to prevent that
                stage2_train_losses = stage2_train_losses[-4000:]
                stage2_val_losses = stage2_val_losses[-1000:]

                self.plot_losses(stage2_train_losses, stage2_val_losses, stage=2, epoch=epoch)

                if epoch % 5 == 0:

                    # Save the checkpoint
                    if checkpoint_path:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scaler_state_dict': self.scaler.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                        }, checkpoint_path.format(epoch, f"stage{stage}"))

    def plot_losses(self, train_losses, val_losses, stage=1, epoch=0):
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
        ax.plot(val_loss_x, val_losses, label="Validation")  # Use the shifted x values for the validation loss curve
        ax.set_title(f"Stage {stage} Epoch {epoch} Losses")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        sns.regplot(x=np.arange(len(train_losses)), y=train_losses, ax=ax, label="Train RegLine", color='blue',
                    scatter=False, order=3)
        sns.regplot(x=val_loss_x, y=val_losses, ax=ax, label="Validation RegLine", color='orange', scatter=False,
                    order=3)
        ax.legend()
        plt.savefig(f"hn_d{stage}_epoch{epoch}.png")
        plt.close()

    def infer_s1_driver(self, dataloder):
        self.model.eval()
        with torch.no_grad():
            for batch, (images_stacked_masked, z, z_vecs, names) in enumerate(dataloder):
                if type(z) == type(None):
                    continue
                images_stacked_masked = images_stacked_masked.to(self.device)
                z = z.to(self.device)

                # Mixed precision validation
                Z = self.model.forward_s1(images_stacked_masked)
                val_loss = self.model.loss_Z(Z, z)

                # print raw outputs for comparison side by side
                for hat, gt in zip(Z, z):
                    print(hat.item(), gt.item())
                print(val_loss)
                break

    def infer_s2_driver(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for batch, (images_stacked_masked, z, z_vecs, names) in enumerate(dataloader):
                if type(z) == type(None):
                    continue
                images_stacked_masked = images_stacked_masked.to(self.device)
                z = z.to(self.device)
                z_vecs = z_vecs.to(self.device)

                # Mixed precision validation
                weights, Z = self.model.forward_s2(images_stacked_masked)
                val_loss = self.model.loss_W(weights, Z, z_vecs, z)
                print(val_loss)
                break

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        # state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'W_head' not in k}
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        return epoch


def s1_train():
    # Modify these lines to use your custom dataloader
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [1, 3, 4, 5]

    train_dataloader, val_dataloader = createDataLoader(coco_path, batchsize=2, channels=channels, num_workers=3,
                                                        shuffle=True)

    model = PoseEstimationModel(len(channels))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True,
                                                           min_lr=0.000001)

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler)
    num_epochs = 20
    save_path = "pose_estimation_model_M_{}_{}.pth"
    trainer.train(num_epochs, checkpoint_path=save_path)


def s2_train():
    # Modify these lines to use your custom dataloader
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [0, 1, 2, 5]

    train_dataloader, val_dataloader = createDataLoader(coco_path, batchsize=8, channels=channels, num_workers=8,
                                                        shuffle=True)

    model = PoseEstimationModel(len(channels))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025, weight_decay=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 25
    scaler = GradScaler()
    max_lr = 0.005
    total_steps = num_epochs * len(train_dataloader)
    pct_start = 0.2  # Percentage of steps for the increasing phase
    anneal_strategy = 'cos'  # Can be 'linear' or 'cos'
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=total_steps,
                                                    pct_start=pct_start, anneal_strategy=anneal_strategy,
                                                    cycle_momentum=True, base_momentum=0.85, max_momentum=0.95)

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler)
    #trainer.load_checkpoint("pose_estimation_model_VT175_5_stage2.pth")
    # trainer.scheduler.pct_start = 0.1


    save_path = "hn_d_{}_{}.pth"

    trainer.train(num_epochs, checkpoint_path=save_path, stage=2)

    torch.save({
        'epoch': 75,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, "hn_d_{}_{}.pth".format(2, 75))



if __name__ == '__main__':
    # s1_train()
    # time.sleep(3600)
    s2_train()
    # inference()
