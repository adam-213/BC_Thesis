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

    def train_one_epoch(self, epoch, splits=16, checkpoint_path=None):
        self.model.train()
        total_loss_rot_list, total_loss_move_list = [], []

        for i, (images, masks, rot, move, names) in enumerate(self.train_dataloader):
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
                            and slicenum < list(range(0, masks[idx].shape[0], slicessize))[-1] \
                            and slicessize > 3:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    print(
                        f"Epoch: {epoch}, Batch: {slicenum + 1}/{masks[idx].shape[0] // slicessize} / {i}/{len(self.train_dataloader)}\
                                             Move Loss: {loss_move_mean.item()}, Rot Loss: {loss_rot_mean.item()}, Cat = {mininames[0]}")

                    if torch.isnan(loss_move_mean) or torch.isnan(loss_rot_mean):
                        print("Nan loss")
                        exit()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % 10 == 0:
                    self.scheduler.step()
                if i % 100 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scaler_state_dict': self.scaler.state_dict(),
                    }, checkpoint_path.format(epoch, i))

        return total_loss_move_list, total_loss_rot_list

    def train(self, num_epochs, checkpoint_path=None):
        for epoch in range(num_epochs):
            train_losses = self.train_one_epoch(epoch, checkpoint_path=checkpoint_path)

            # Save the checkpoint
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'train_losses': train_losses,
                }, checkpoint_path.format(epoch, "full"))


def main():
    # Modify these lines to use your custom dataloader
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [0, 1, 2, 3, 4, 5, 9]

    train_dataloader, val_dataloader = createDataLoader(coco_path, batchsize=4, channels=channels, num_workers=6,
                                                        shuffle=True)

    model = PoseEstimationModel(len(channels) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, device, scaler, scheduler)
    num_epochs = 50
    save_path = "pose_estimation_model_{}_{}.pth"
    trainer.train(num_epochs, checkpoint_path=save_path)


if __name__ == '__main__':
    main()
