import time

import numpy

from b_DataLoader_RCNN import createDataLoader

from c_MaskRCNN import MaskRCNN as rcModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pathlib
from torchvision.ops import box_iou
import json
from pycocotools import mask as mask_util
from tqdm import tqdm


def geometric(pointcloud, mask):
    masked = pointcloud * mask
    # filter out the zeros
    reshaped = masked.reshape(3, -1)
    filter_out = np.any(reshaped, axis=0)
    filtered = reshaped[:, filter_out]

    # get the mean
    mean = np.mean(filtered, axis=1)

    return mean


# Set up the camera intrinsic parameters
intrinsics = {
    'fx': 1181.077335,
    'fy': 1181.077335,
    'cx': 516.0,
    'cy': 386.0
}


def world_to_image_coords(world_coords, intrinsics):
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    X, Y, Z = world_coords

    # Normalize the real-world coordinates
    x = X / Z
    y = Y / Z

    # Apply the intrinsic parameters to convert to pixel coordinates
    u = fx * x + cx
    v = fy * y + cy

    # Round to integer values
    u, v = round(u), round(v)

    return u, v


base_path = pathlib.Path(__file__).parent.absolute()
coco_path = base_path.joinpath('RevertDS')
channels = [0, 1, 2, 5, 9]

full_loader, stats = createDataLoader(coco_path, bs=1,
                                      num_workers=0,
                                      channels=channels,
                                      shuffle=False,
                                      dataset_creation=True)
mean, std = stats
dataset = full_loader.dataset
model = rcModel(5, 5, mean[channels], std[channels])

checkpoint = torch.load("RCNN_Unscaled_2cat24.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.cuda()
model.eval()

coco_dict = {"images": [], "annotations": []}
annotation_id = 1

for i, (images, target, pointcloud) in enumerate(tqdm(full_loader)):
    # extract the image id and get the file name by id from the dataset
    # done to ensure consistency with the coco dataset and images
    image_id = dataset.coco.imgs[i]["id"]
    file_name = dataset.coco.imgs[image_id]["file_name"]
    height = dataset.coco.imgs[image_id]["height"]
    width = dataset.coco.imgs[image_id]["width"]
    lic = dataset.coco.imgs[image_id]["license"]
    # print(file_name)
    # add the image to the new coco dict
    coco_dict["images"].append(
        {"id": image_id, "file_name": file_name, "height": height, "width": width, "license": lic})

    # predict
    images = images.cuda().float()
    outputs = model(images)

    # break down the outputs
    boxes = outputs[0]['boxes'].cpu().detach().numpy()
    labels = outputs[0]['labels'].cpu().detach().numpy()
    scores = outputs[0]['scores'].cpu().detach().numpy()
    masks = outputs[0]['masks'].cpu().detach().numpy()

    # break down the targets
    tboxes = target[0]['boxes'].cpu().detach().numpy()
    tlabels = target[0]['labels'].cpu().detach().numpy()
    tmasks = target[0]['masks'].cpu().detach().numpy()
    ttms = target[0]['tm'].cpu().detach().numpy()
    tarea = target[0]['area'].cpu().detach().numpy()

    # match the outputs to the targets with mask iou
    for box, mask, label, score in zip(boxes, masks, labels, scores):
        # find the best match
        iou = box_iou(torch.tensor(box).unsqueeze(0), torch.tensor(tboxes)).squeeze(0).numpy()
        best = np.argmax(iou)
        # get the best match
        tbox = tboxes[best]
        tmask = tmasks[best]
        tlabel = tlabels[best]
        ttm = ttms[best]
        # calculate the mask iou
        iou = np.sum(mask * tmask) / np.sum(mask + tmask - mask * tmask)
        if iou > 0.75 and label >= 3:  # filter out background and bins they are sorted so this is fine
            # the power of maskrcnn produces matches at worst 0.9 iou
            # print("Matched")
            mask = mask[0, :, :]
            # treshold the mask otherwise it's a rectangle for some reason /shrug
            mask = mask > 0.8
            rle_mask = mask_util.encode(np.asfortranarray(mask.astype(bool)))
            rle_box = mask_util.toBbox(rle_mask)
            rle_area = mask_util.area(rle_mask)
            # look up the label name
            name = dataset.coco.cats[label]['name']

            # # compute the centroid of the mask with moments # OLD
            # from copy import deepcopy
            #
            # maskd = deepcopy(mask)
            # maskint = (maskd * 255).astype(np.uint8)
            # maskthresh = cv2.threshold(maskint, 0, 240, cv2.THRESH_BINARY)[1]
            # moments = cv2.moments(maskthresh)
            #
            # # Compute the centroid using the moments
            # if moments["m00"] != 0:
            #     centroid_x = int(moments["m10"] / moments["m00"])
            #     centroid_y = int(moments["m01"] / moments["m00"])
            #     # print("Centroid: ({}, {})".format(centroid_x, centroid_y))
            #
            #     # look up the depth from the depth image
            #     depth_image = images[0, 3, :, :].cpu().detach().numpy()
            #     depth_value = depth_image[centroid_y, centroid_x]
            # else:
            #     raise ValueError("Centroid is 0,0")

            # replace with geometric centroid of the pointcloud

            centroid = geometric(pointcloud[0].cpu().detach().numpy(), mask)
            centroid_x = centroid[0]
            centroid_y = centroid[1]
            depth_value = centroid[2]

            # check if the transform is below the threshold for a match

            gt_centroid = ttm[0:3, 3]
            # compute mae of the centroid
            centroid_mae = np.mean(np.abs(centroid - gt_centroid))

            if centroid_mae > 100:  # threshold for a match so we dont match another mask to the same object
                print("Centroid MAE: {}".format(centroid_mae), gt_centroid, centroid)
                continue

            centroid_x, centroid_y = world_to_image_coords(centroid, intrinsics)


            annotation = {
                "id": int(label),
                "category_id": int(label),
                "image_id": int(image_id),
                "bbox": rle_box.tolist(),
                "score": float(score),
                "mask": rle_mask['counts'].decode('utf-8'),
                "size": rle_mask['size'],
                "area": float(rle_area),
                "transform": ttm.flatten(order='F').tolist(),
                "name": name,
                "centroid": [float(centroid_x), float(centroid_y), float(depth_value)],
            }
            coco_dict["annotations"].append(annotation)
            annotation_id += 1
            # # plot the results, debug
            # fig, (ax1, ax2) = plt.subplots(1, 2)
            #
            # # convert the image back to the original form
            # img = images[0].cpu().permute(1, 2, 0).detach().numpy()
            # img = img.astype(np.uint8)
            # img = img[:, :, :3]
            #
            # # plot the image with ground truth mask
            # ax1.imshow(img)
            # ax1.imshow(tmask, alpha=0.7)
            # ax1.set_title(f"Ground Truth: Label {tlabel}")
            #
            # # plot the image with predicted mask
            # masker = mask_util.decode(rle_mask)
            # ax2.imshow(img)
            # ax2.imshow(masker, alpha=0.7)
            # ax2.set_title(f"Prediction: Label {label}, Score {score:.2f}")
            #
            # plt.show()

with open(coco_path.joinpath("annotations", "merged.json"), 'r') as f:
    j = json.load(f)

j['images'] = coco_dict['images']
j['annotations'] = coco_dict['annotations']
with open(coco_path.joinpath("annotations", "merged_maskrcnn_centroid2.json"), 'w') as f:
    json.dump(j, f)