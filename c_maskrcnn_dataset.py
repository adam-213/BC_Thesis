import time

import numpy

from b_DataLoader_RCNN import createDataLoader

from b_MaskRCNN import MaskRCNN

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
if __name__ == '__main__':
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCOFULL_Dataset')
    channels = [0, 1, 2, 5, 9]
    # use all channels to get full representation for depth estimation
    # but only pass the select channels to the model
    full_loader, stats = createDataLoader(coco_path, bs=1,
                                          num_workers=1,
                                          channels=None,
                                          shuffle=False,
                                          dataset_creation=True)
    mean, std = stats
    dataset = full_loader.dataset
    cats = full_loader.dataset.coco.cats
    model = MaskRCNN(5, len(cats), mean[channels], std[channels])

    checkpoint = torch.load("RCNN_Unscaled_34.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()

    coco_dict = {"images": [], "annotations": []}
    annotation_id = 1

    for i, (fullimages, target) in enumerate(tqdm(full_loader)):
        images = fullimages.clone()[:, channels, :, :]  # only pass the selected channels
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
            if label <= 2:
                continue
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
            if iou > 0.84:
                # the power of maskrcnn produces matches at worst 0.9 iou
                # print("Matched")
                mask = mask[0, :, :]
                # treshold the mask otherwise it's a rectangle for some reason /shrug
                mask = mask > 0.75
                rle_mask = mask_util.encode(np.asfortranarray(mask.astype(bool)))
                rle_box = mask_util.toBbox(rle_mask)
                rle_area = mask_util.area(rle_mask)
                # look up the label name
                name = dataset.coco.cats[label]['name']

                # compute the centroid of the mask with moments
                from copy import deepcopy

                maskd = deepcopy(mask)
                maskint = (maskd * 255).astype(np.uint8)
                maskthresh = cv2.threshold(maskint, 0, 240, cv2.THRESH_BINARY)[1]
                moments = cv2.moments(maskthresh)

                if moments["m00"] != 0:
                    centroid_x = int(moments["m10"] / moments["m00"])
                    centroid_y = int(moments["m01"] / moments["m00"])

                    # Extract depth, point cloud, normals, and albedo from the images array
                    depth_image = fullimages[0, 3, :, :].cpu().detach().numpy() * mask
                    point_cloud = fullimages[0, 3:6, :, :].cpu().detach().numpy()* mask
                    normals = fullimages[0, 6:9, :, :].cpu().detach().numpy()* mask
                    albedo = fullimages[0, 9, :, :].cpu().detach().numpy()* mask

                    # Reshape the point cloud, normals, and albedo to 3 x N arrays
                    point_cloud = point_cloud.reshape(3, -1)
                    normals = normals.reshape(3, -1)
                    albedo = albedo.reshape(1, -1)

                    # Compute the dot product of normals and viewing direction
                    viewing_direction = np.array([0, 0, 1]).reshape(3, 1)
                    normal_weights = np.dot(normals.T, viewing_direction)

                    # Incorporate albedo information into the weights (optional)
                    # Here, we just multiply the normal_weights by the albedo
                    combined_weights = normal_weights * albedo.T

                    # Compute the weighted average of the point cloud
                    weighted_sum = np.dot(point_cloud, combined_weights)
                    total_weights = np.sum(combined_weights)
                    weighted_centroid = weighted_sum / total_weights

                    print("Weighted Centroid: ({}, {}, {})".format(*weighted_centroid.flatten()))
                    centroid_x, centroid_y, depth_value = weighted_centroid.flatten()

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
                    # # plot the results
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

                else:
                    raise ValueError("Centroid is 0,0")
            else:
                print("Not Matched", iou)
                # # visualize the results
                # # plot the results
                # fig, ax = plt.subplots(1, 1)
                # img = images[0, :3]
                # img = img.cpu().detach().permute(1, 2, 0).numpy()
                # ax.imshow(img)
                # ax.imshow(mask.transpose(1, 2, 0), alpha=0.5, cmap='jet')
                # ax.imshow(tmask, alpha=0.5, cmap="cool")
                # plt.show()
                # print("Predicted", box, label, score)

    cocopath = base_path.joinpath('COCOFULL_Dataset', 'annotations')
    with open(cocopath.joinpath("merged.json"), 'r') as f:
        j = json.load(f)

    j['images'] = coco_dict['images']
    j['annotations'] = coco_dict['annotations']
    with open(cocopath.joinpath("merged_maskrcnn_centroid.json"), 'w') as f:
        json.dump(j, f)
