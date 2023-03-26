import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pathlib

from b_DataLoader_RCNN import createDataLoader
from c_MaskRCNN import MaskRCNN


def display_image_and_mask(image, mask, box, threshold=0.1):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    img = image[:, :, :3].copy()
    ax[0].imshow(img)

    overlay = np.zeros_like(img)
    overlay[mask > threshold] = [1, 0, 0]
    img = cv2.addWeighted(img, 1, overlay, 0.5, 0)

    x1, y1, x2, y2 = box
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='g', linewidth=2)
    ax[1].add_patch(rect)
    ax[1].imshow(img)

    plt.show()


def visualize_output(images, targets, outputs, threshold=0.1):
    for i in range(len(images)):
        image = images[i].cpu().detach().numpy().transpose(1, 2, 0)
        target = targets[i]
        output = outputs[i]

        masks = output["masks"].cpu().detach().numpy().squeeze()
        scores = output["scores"].cpu().detach().numpy().squeeze()
        boxes = output["boxes"].cpu().detach().numpy().squeeze()

        sorted_indices = scores.argsort()[::-1]
        masks = masks[sorted_indices]
        boxes = boxes[sorted_indices]

        for mask, box in zip(masks, boxes):
            display_image_and_mask(image, mask, box, threshold=threshold)


if __name__ == '__main__':
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [0, 1, 2, 5, 9]

    train_dataloader, val_dataloader, stats = createDataLoader(coco_path, bs=1, num_workers=0, channels=channels)
    mean, std = stats
    mean, std = mean[channels], std[channels]

    model = MaskRCNN(5, 5, mean, std)
    model.cuda()
    model.eval()

    checkpoint = torch.load("RCNN_TM_18.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    images, targets = next(iter(val_dataloader))

    images = images.cuda().float()

    outputs = model(images)

    images_cpu = images.cpu()
    visualize_output(images_cpu, targets, outputs, threshold=0.2)
