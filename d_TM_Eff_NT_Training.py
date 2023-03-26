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

    def train_one_epoch(self, epoch, batchsize=2, checkpoint_path=None):
        #
        self.model.train()
        total_loss_rot_list, total_loss_move_list = [], []

        for i, (images, masks, rot, move, names) in enumerate(self.train_dataloader):
            for idx, image in enumerate(images):
                for slicenum, sliceidx in enumerate(range(0, masks[idx].shape[0], batchsize)):
                    slicesize = min(batchsize, masks[idx].shape[0] - batchsize * slicenum)
                    # Cut the microbatch
                    minimasks = masks[idx][sliceidx:sliceidx + slicesize]
                    minirot = rot[idx][sliceidx:sliceidx + slicesize]
                    minimove = move[idx][sliceidx:sliceidx + slicesize]
                    mininames = names[idx][sliceidx:sliceidx + slicesize]
                    # remove the first dimension
                    minimasks.squeeze_(0)
                    minirot.squeeze_(0)
                    minimove.squeeze_(0)

                    # stack the image to the same size as the masks
                    if len(minimasks.shape) != 3:
                        # Workaround in case the microbatch size is 1
                        minimasks = minimasks.unsqueeze(0)
                        minirot = minirot.unsqueeze(0)
                        minimove = minimove.unsqueeze(0)

                    # copy the image to match the microbatch size
                    image_stacked = torch.stack([image] * minimasks.shape[0])
                    # stack minimasks to the image as a channel
                    # image_masks_stacked = torch.cat((image_stacked, minimasks.unsqueeze(1)), dim=1)
                    images_rgb, images_aux = image_stacked[:, :3], image_stacked[:, 3:]
                    images_rgb_masked = torch.cat((images_rgb, minimasks.unsqueeze(1)), dim=1)
                    images_aux_masked = images_aux * minimasks.unsqueeze(1)
                    del image_stacked, images_rgb, images_aux

                    image_masks_stacked = torch.cat((images_rgb_masked, images_aux_masked), dim=1)
                    del images_rgb_masked, images_aux_masked
                    # move to device
                    image_masks_stacked = image_masks_stacked.to(self.device)
                    minirot = minirot.to(self.device)
                    minimove = minimove.to(self.device)
                    # mininames = mininames.to(self.device)

                    # turn on grad for input
                    image_masks_stacked.requires_grad_(True)

                    self.optimizer.zero_grad()

                    # with autocast():
                    movehat, rothat = checkpoint(self.model, image_masks_stacked)
                    loss_move, loss_rot = self.model.loss(movehat, rothat, minimove, minirot, mininames)
                    loss_move_mean, loss_rot_mean = loss_move.mean(), loss_rot.mean()
                    # Compute the gradients for each loss separately
                    del loss_move, loss_rot, image_masks_stacked, minimasks, minirot, minimove

                    loss_move_mean.backward(retain_graph=True)
                    loss_rot_mean.backward()

                    # Scale the gradients
                    # self.scaler.scale(loss_move_mean)
                    # self.scaler.scale(loss_rot_mean)
                    # Update the optimizer with the combined gradients
                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()
                    total_loss_rot_list.append(loss_rot_mean.item())
                    total_loss_move_list.append(loss_move_mean.item())

                    if random.random() < 0.25 \
                            and slicenum < list(range(0, masks[idx].shape[0], slicesize))[-1] \
                            and slicesize > 3:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    print(
                        f"Epoch: {epoch}, Batch: {slicenum + 1}/{masks[idx].shape[0] // slicesize} / {i}/{len(self.train_dataloader)}\
                        Move Loss: {loss_move_mean.item()}, Rot Loss: {loss_rot_mean.item()},\
                        Cat = {mininames[0]}")

                    if torch.isnan(loss_move_mean) or torch.isnan(loss_rot_mean):
                        print("Nan loss")
                        exit()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % 10 == 0:
                    self.scheduler.step()
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(total_loss_move_list, label="Move Loss")
                    ax.plot(total_loss_rot_list, label="Rot Loss")
                    ax.legend()
                    ax.set_title("Loss")
                    plt.savefig(f"losses_{epoch}_{i}.png")
                    plt.close()

                if i % 100 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scaler_state_dict': self.scaler.state_dict(),
                    }, checkpoint_path.format(epoch, i))

        return total_loss_move_list, total_loss_rot_list

    def train(self, num_epochs, checkpoint_path=None):
        train_losses_move, train_losses_rot = [], []
        val_losses_move, val_losses_rot = [], []

        for epoch in range(num_epochs):
            train_loss_move, train_loss_rot = self.train_one_epoch(epoch, checkpoint_path=checkpoint_path)
            train_losses_move.append(train_loss_move)
            train_losses_rot.append(train_loss_rot)

            val_loss_move, val_loss_rot = self.validate_one_epoch(epoch)
            val_losses_move.append(val_loss_move)
            val_losses_rot.append(val_loss_rot)

            self.plot_losses(train_losses_move, train_losses_rot, val_losses_move, val_losses_rot)

            # Save the checkpoint
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'train_losses_move': train_losses_move,
                    'train_losses_rot': train_losses_rot,
                    'val_losses_move': val_losses_move,
                    'val_losses_rot': val_losses_rot,
                }, checkpoint_path.format(epoch, "full"))

    def validate_one_epoch(self, epoch, splits=16):
        self.model.eval()
        total_val_loss_rot_list, total_val_loss_move_list = [], []

        with torch.no_grad():
            for i, (images, masks, rot, move, names) in enumerate(self.val_dataloader):
                for idx, image in enumerate(images):
                    try:
                        slicessize = max(1, masks[idx].shape[0] // splits)
                    except:
                        print(masks[idx].shape)
                        slicessize = 1
                    for slicenum, slice in enumerate(range(0, masks[idx].shape[0], slicessize)):
                        # Cut the microbatch
                        minimasks = masks[idx][slice:slice + slicessize]
                        minirot = rot[idx][slice:slice + slicessize]
                        minimove = move[idx][slice:slice + slicessize]
                        mininames = names[idx][slice:slice + slicessize]
                        # remove the first dimension
                        minimasks.squeeze_(0)
                        minirot.squeeze_(0)
                        minimove.squeeze_(0)

                        # stack the image to the same size as the masks
                        if len(minimasks.shape) != 3:
                            # Workaround in case the microbatch size is 1
                            minimasks = minimasks.unsqueeze(0)
                            minirot = minirot.unsqueeze(0)
                            minimove = minimove.unsqueeze(0)

                        # copy the image to match the microbatch size
                        image_stacked = torch.stack([image] * minimasks.shape[0])
                        # stack minimasks to the image as a channel
                        # image_masks_stacked = torch.cat((image_stacked, minimasks.unsqueeze(1)), dim=1)
                        images_rgb, images_aux = image_stacked[:, :3], image_stacked[:, 3:]
                        images_rgb_masked = torch.cat((images_rgb, minimasks.unsqueeze(1)), dim=1)
                        images_aux_masked = images_aux * minimasks.unsqueeze(1)
                        del image_stacked, images_rgb, images_aux

                        image_masks_stacked = torch.cat((images_rgb_masked, images_aux_masked), dim=1)
                        del images_rgb_masked, images_aux_masked
                        # move to device
                        image_masks_stacked = image_masks_stacked.to(self.device)
                        minirot = minirot.to(self.device)
                        minimove = minimove.to(self.device)
                        # mininames = mininames.to(self.device)

                        # turn on grad for input
                        image_masks_stacked.requires_grad_(True)

                        self.optimizer.zero_grad()

                        # with autocast():
                        movehat, rothat = checkpoint(self.model, image_masks_stacked)
                        loss_move, loss_rot = self.model.loss(movehat, rothat, minimove, minirot, mininames)
                        loss_move_mean, loss_rot_mean = loss_move.mean(), loss_rot.mean()
                        # Compute the gradients for each loss separately
                        del loss_move, loss_rot, image_masks_stacked, minimasks, minirot, minimove
                total_val_loss_rot_list.append(loss_rot_mean.item())
                total_val_loss_move_list.append(loss_move_mean.item())

        avg_val_loss_rot = np.mean(total_val_loss_rot_list)
        avg_val_loss_move = np.mean(total_val_loss_move_list)
        return avg_val_loss_move, avg_val_loss_rot

    def plot_losses(self, train_losses_move, train_losses_rot, val_losses_move, val_losses_rot):
        clear_output(wait=True)
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses_move, label="Train Move Loss")
        plt.plot(val_losses_move, label="Validation Move Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_losses_rot, label="Train Rot Loss")
        plt.plot(val_losses_rot, label="Validation Rot Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()

    def inference_bactch(self, epoch, batchsize=1, checkpoint_path=None):
        #
        self.model.eval()
        total_loss_rot_list, total_loss_move_list = [], []

        for i, (images, masks, rot, move, names) in enumerate(self.train_dataloader):
            for idx, image in enumerate(images):
                for slicenum, sliceidx in enumerate(range(0, masks[idx].shape[0], batchsize)):
                    slicesize = min(batchsize, masks[idx].shape[0] - batchsize * slicenum)
                    # Cut the microbatch
                    minimasks = masks[idx][sliceidx:sliceidx + slicesize]
                    minirot = rot[idx][sliceidx:sliceidx + slicesize]
                    minimove = move[idx][sliceidx:sliceidx + slicesize]
                    mininames = names[idx][sliceidx:sliceidx + slicesize]
                    # remove the first dimension
                    minimasks.squeeze_(0)
                    minirot.squeeze_(0)
                    minimove.squeeze_(0)

                    # stack the image to the same size as the masks
                    if len(minimasks.shape) != 3:
                        # Workaround in case the microbatch size is 1
                        minimasks = minimasks.unsqueeze(0)
                        minirot = minirot.unsqueeze(0)
                        minimove = minimove.unsqueeze(0)

                    # copy the image to match the microbatch size
                    image_stacked = torch.stack([image] * minimasks.shape[0])
                    # stack minimasks to the image as a channel
                    # image_masks_stacked = torch.cat((image_stacked, minimasks.unsqueeze(1)), dim=1)
                    images_rgb, images_aux = image_stacked[:, :3], image_stacked[:, 3:]
                    images_rgb_masked = torch.cat((images_rgb, minimasks.unsqueeze(1)), dim=1)
                    images_aux_masked = images_aux * minimasks.unsqueeze(1)
                    del image_stacked, images_rgb, images_aux

                    image_masks_stacked = torch.cat((images_rgb_masked, images_aux_masked), dim=1)
                    del images_rgb_masked, images_aux_masked
                    # move to device
                    image_masks_stacked = image_masks_stacked.to(self.device)
                    minirot = minirot.to(self.device)
                    minimove = minimove.to(self.device)
                    # mininames = mininames.to(self.device)

                    # with autocast():
                    movehat, rothat = self.model(image_masks_stacked)
                    loss_move, loss_rot = self.model.loss(movehat, rothat, minimove, minirot, mininames)
                    loss_move_mean, loss_rot_mean = loss_move.mean(), loss_rot.mean()
                    np.set_printoptions(suppress=True, precision=5)
                    print("Pred", "Move", movehat.detach().cpu().numpy(), "Rot", rothat.detach().cpu().numpy())
                    print("GT", "Move", minimove.detach().cpu().numpy(), "Rot", minirot.detach().cpu().numpy())
                    print("Loss", "Move", loss_move_mean.item(), "Rot", loss_rot_mean.item())

            break

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])


def main():
    # Modify these lines to use your custom dataloader
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [0, 1, 2, 5]

    train_dataloader, val_dataloader = createDataLoader(coco_path, batchsize=4, channels=channels, num_workers=6,
                                                        shuffle=True)

    model = PoseEstimationModel(len(channels) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler)
    num_epochs = 50
    save_path = "pose_estimation_model_{}_{}.pth"
    trainer.train(num_epochs, checkpoint_path=save_path)


def inference():
    # Modify these lines to use your custom dataloader
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [0, 1, 2, 5]

    train_dataloader, val_dataloader = createDataLoader(coco_path, batchsize=4, channels=channels, num_workers=3,
                                                        shuffle=True)

    model = PoseEstimationModel(len(channels) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler)
    trainer.load_checkpoint("pose_estimation_model_0_full.pth")
    trainer.inference_bactch(1, batchsize=1, checkpoint_path=None)


if __name__ == '__main__':
    main()
    # inference()
