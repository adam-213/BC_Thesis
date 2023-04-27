import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pathlib

from b_DataLoader_RCNN import createDataLoader
from b_MaskRCNN import MaskRCNN


def display_image_and_mask(image, mask, box, threshold=0.1):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    img = image[:, :, :3].copy()
    ax[0].imshow(img)

    overlay = np.zeros_like(img)
    overlay[mask > threshold] = [1, 0, 0]
    img = cv2.addWeighted(img, 1, overlay, 0.5, 0)

    x1, y1, x2, y2 = box
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='g', linewidth=2)

    # Compute the moments of the binary mask
    maskint = (mask * 256).astype(np.uint8)
    maskthresh = cv2.threshold(maskint, 0, 240, cv2.THRESH_BINARY)[1]
    moments = cv2.moments(maskthresh)

    # Compute the centroid using the moments
    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])
    print("Centroid: ({}, {})".format(centroid_x, centroid_y))

    # Draw the centroid in the result image
    ax[0].scatter(centroid_x, centroid_y, c='g', s=50, marker='o')
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

        # if mask area is more than say 50% of the image, then remove it
        # this is to remove the background mask
        areas = [mask[mask != 0].sum() for mask in masks]
        areas = np.array(areas)
        areas = areas < 0.2 * image.shape[0] * image.shape[1]
        masks = masks[areas]
        boxes = boxes[areas]


        # masks = masks[:3]
        # boxes = boxes[:3]

        for mask, box, area in zip(masks, boxes, areas):
            print("Mask area: {}".format(area))
            display_image_and_mask(image, mask, box, threshold=threshold)


if __name__ == '__main__':
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCOFULL_Dataset')
    channels = [0, 1, 2, 5, 9]

    train_dataloader, val_dataloader, stats = createDataLoader(coco_path, bs=1, num_workers=0,
                                                               channels=channels)
    mean, std = stats
    mean, std = mean[channels], std[channels]

    model = MaskRCNN(5, 6, mean, std)
    # model.cuda()
    model.eval()

    checkpoint = torch.load("RCNN_Unscaled_19.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    images, targets = next(iter(val_dataloader))

    # images = images.cuda().float()


    outputs = model(images)

    images_cpu = images.cpu()
    visualize_output(images_cpu, targets, outputs, threshold=0.2)
