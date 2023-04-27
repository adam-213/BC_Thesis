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
from d_TM_Eff_NT import PoseEstimationModel


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
        total_loss_list = []
        for batch, (images_stacked_masked, z_vecs, XYZs) in enumerate(self.train_dataloader):
            if type(z_vecs) == type(None):
                continue
            self.optimizer.zero_grad()
            images_stacked_masked = images_stacked_masked.to(self.device)
            z_vecs = z_vecs.to(self.device)
            XYZs = XYZs.to(self.device)

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
            # Update the optimizer with the combined gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
            for batch, (images_stacked_masked, z_vecs, XYZs) in enumerate(self.val_dataloader):
                print(time.time())
                if type(z_vecs) == type(None):
                    continue
                self.optimizer.zero_grad()
                images_stacked_masked = images_stacked_masked.to(self.device)
                z_vecs = z_vecs.to(self.device)
                XYZs = XYZs.to(self.device)
                plot = True if batch == self.plotbatch else False
                # Mixed precision training for speeeeed
                with autocast():
                    weights = self.model(images_stacked_masked, XYZs)
                    val_loss = self.model.loss_W(weights, z_vecs, plot, images_stacked_masked, XYZs, [batch, epoch])

                total_val_loss_list.append(val_loss.item())

                print(f"Validation Epoch {epoch} Batch {batch} Loss: {val_loss.item()}")

        self.plotbatch += 1
        return total_val_loss_list

    def train(self, num_epochs, stage=1, checkpoint_path=None):
        stage1_train_losses, stage1_val_losses = [], []
        stage2_train_losses, stage2_val_losses = [], []
        self.colors = sns.color_palette("husl", num_epochs)

        for epoch in range(num_epochs):
            s2_loss = self.train_one_epoch_stage_2(epoch)

            stage2_train_losses.append(s2_loss)

            s2_loss_val = self.val_one_epoch_stage_2(epoch)
            stage2_val_losses.append(s2_loss_val)
            # stage2_val_losses.append(s2_loss)
            # I had a crash on memory here so this is to prevent that
            # keep last 4 epochs
            stage2_train_losses = stage2_train_losses[-4:]
            # stage2_val_losses = stage2_val_losses[-4:]
            # self.scheduler.update_lr_on_plateau(np.mean(s2_loss_val))
            self.scheduler.step(np.mean(s2_loss_val))
            t = time.time()
            self.plot_losses(stage2_train_losses, stage2_val_losses, stage=2, epoch=epoch)
            print("plotting took", time.time() - t)
            if epoch % 2 == 0:

                # Save the checkpoint
                if checkpoint_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scaler_state_dict': self.scaler.state_dict(),
                        # 'scheduler_state_dict': self.scheduler.state_dict(),
                    }, checkpoint_path.format(epoch, f"stage{stage}"))

    def plot_losses(self, train_losses, val_losses, stage=2, epoch=0):
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
        plt.savefig(f"deit3_epoch{epoch}.png")
        plt.close()

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

class DampedCosineScheduler:
    def __init__(self, optimizer, base_lr, max_lr, base_decay_factor, decay_factor, steps_per_epoch, num_epochs):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.base_decay_factor = base_decay_factor
        self.decay_factor = decay_factor
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs
        self.current_step = 0

        # Initialize the ReduceLROnPlateau scheduler
        self.reduce_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                                         patience=2,
                                                                         verbose=True, threshold=1e-4,
                                                                         threshold_mode='rel', cooldown=0, min_lr=0,
                                                                         eps=1e-8)

    def step(self):
        self.current_step += 1
        epoch = self.current_step // self.steps_per_epoch
        step_in_epoch = self.current_step % self.steps_per_epoch

        # Calculate the current high value of the learning rate
        current_max_lr = self.max_lr * (self.decay_factor ** epoch)

        # Calculate the current low value of the learning rate
        current_base_lr = self.base_lr * (self.base_decay_factor ** epoch)

        # Calculate the learning rate using a cosine transition between the high and low values
        lr = current_base_lr + 0.5 * (current_max_lr - current_base_lr) * (
                1 + np.cos(np.pi * step_in_epoch / self.steps_per_epoch))

        # Update the learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update_lr_on_plateau(self, val_loss):
        # Simulate the call to step() with the current validation loss
        self.reduce_plateau.step(val_loss)

        # Update the base_lr and max_lr with the reduced learning rate
        factor = self.reduce_plateau.factor
        print(f"Reducing learning rate by a factor of {factor}")
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            new_lr = current_lr * factor
            self.base_lr = self.base_lr * factor
            self.max_lr = self.max_lr * factor
            param_group['lr'] = new_lr

def s2_train():
    # Modify these lines to use your custom dataloader
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCOFULL_Dataset')
    channels = [0, 1, 2, 5, 9]
    gray = True

    train_dataloader, val_dataloader = createDataLoader(coco_path, batchsize=3, channels=channels, num_workers=8,
                                                        shuffle=True, gray=gray)

    # model = PoseEstimationModel(len(channels) - 2 if gray else len(channels))
    model = PoseEstimationModel(len(channels) - 2 if gray else len(channels))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 50
    scaler = GradScaler()

    # Set the base and max learning rates
    base_lr = 0.00009

    optimizer = torch.optim.RAdam(model.parameters(), lr=base_lr, weight_decay=0.0001)

    # Initialize the DampedCosineScheduler with decay factors for both base and max learning rates
    # scheduler = DampedCosineScheduler(optimizer, base_lr=base_lr, max_lr=max_lr,
    #                                   decay_factor=0.5, base_decay_factor=0.7,
    #                                   steps_per_epoch=len(train_dataloader), num_epochs=num_epochs)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2,
                                                           verbose=True, threshold=1e-3,
                                                           threshold_mode='rel', cooldown=0, min_lr=0,
                                                           eps=1e-8)
    #
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001, last_epoch=-1)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.00025,
    #                                                 steps_per_epoch=len(train_dataloader),
    #                                                 epochs=num_epochs, anneal_strategy='cos', cycle_momentum=False,
    #                                                 base_momentum=0.85, max_momentum=0.95, div_factor=15.0,
    #                                                 final_div_factor=10000.0, last_epoch=-1, verbose=False,
    #                                                 pct_start=0.3)

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler)
    # trainer.load_checkpoint("pose_estimation_model_VT175_5_stage2.pth")
    # trainer.scheduler.pct_start = 0.1

    save_path = "Unscaled_{}_{}.pth"

    trainer.train(num_epochs, checkpoint_path=save_path, stage=2)

    torch.save({
        'epoch': "full",
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, "full50.pth")

if __name__ == '__main__':
    # s1_train()
   #time.sleep(3000)
    s2_train()
    # inference()
