import time

import numpy

from b_DataLoader_RCNN import createDataLoader as rcdataloader

from c_TM_Eff_NT import PoseEstimationModel as peModel
from c_MaskRCNN import MaskRCNN as rcModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

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


def image_to_world_coords(image_coords, intrinsics, Z):
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    u, v = image_coords

    # Convert pixel coordinates to normalized image coordinates
    x = (u - cx) / fx
    y = (v - cy) / fy

    # Compute the real-world coordinates
    X = x * Z
    Y = y * Z

    return X, Y, Z


def prepare_inference():
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    channels = [0, 1, 2, 3, 4, 5, 9]
    chans_sel = [0, 1, 2, 5, 6]
    # create unshuffled dataloaders
    rcdata, val_dataloader, stats = rcdataloader(coco_path, 1, channels=channels, shuffle=True, num_workers=0)
    mean, std = stats
    mean, std = np.array(mean), np.array(std)
    # cut out the channels we don't need
    sel = np.array(channels)[chans_sel]
    mean, std = mean[sel], std[sel]

    rcmodel = rcModel(5, 5, mean, std)

    # create models
    tmmodel = peModel(2)

    # load weights
    tmmodel.load_state_dict(torch.load('full.pth')['model_state_dict'])
    rcmodel.load_state_dict(torch.load(base_path.joinpath("rcnn", "RCNN_TM_18.pth"))['model_state_dict'])

    # set models to eval mode
    tmmodel.eval()
    rcmodel.eval()

    # set device to cpu
    device = torch.device('cpu')
    tmmodel.to(device)
    rcmodel.to(device)

    return rcmodel, tmmodel, rcdata, device


def infer_mrcnn(model, image):
    rcoutputs = model(image)
    # argsort by confidence
    if len(rcoutputs) >= 1:
        mask = rcoutputs[0]["masks"]
        labels = rcoutputs[0]["labels"]
        scores = rcoutputs[0]["scores"]
        boxes = rcoutputs[0]["boxes"]

        mask = mask[labels > 2]
        scores = scores[labels > 2]
        boxes = boxes[labels > 2]
        labels = labels[labels > 2]

        # sort by confidence
        sort = torch.argsort(scores, descending=True)
        mask = mask[sort]
        labels = labels[sort]
        scores = scores[sort]
        boxes = boxes[sort]

        # filter out labels 0,1,2 - backround and boxes, don't care about them, for all intents and purposes immovable
        mask = mask[labels > 2]
        scores = scores[labels > 2]
        boxes = boxes[labels > 2]
        labels = labels[labels > 2]

        if len(scores) > 0:
            # get the best one

            best = [{"masks": mask[0], "labels": labels[0], "scores": scores[0], "boxes": boxes[0]}]
            return best
        else:
            print("No objects detected_besides_bin,rly?")

    else:
        print("No objects detected,rly?")

    plt.imshow(image[0, :3, :, :].permute(1, 2, 0).detach().numpy())
    plt.show()


from copy import deepcopy


def translation_layer(best, image):
    # Prepare data for pose estimation
    # get bbox
    bbox = best[0]['boxes'].detach().numpy()
    # get mask
    mask = best[0]['masks'].detach().numpy()

    # Compute the moments of the binary mask
    maskd = deepcopy(mask).transpose(1, 2, 0)
    maskint = (maskd * 256).astype(np.uint8)
    maskthresh = cv2.threshold(maskint, 0, 240, cv2.THRESH_BINARY)[1]
    moments = cv2.moments(maskthresh)

    # Compute the centroid using the moments
    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
        print("Centroid: ({}, {})".format(centroid_x, centroid_y))

        # look up the depth from the depth image
        depth_image = image[0, 3, :, :]
        depth_value = depth_image[centroid_y, centroid_x]
    else:
        print("No valid centroid found.")
        centroid_x, centroid_y, depth_value = None, None, None
    x1, y1, x2, y2 = bbox

    # expand the bbox by k
    k = 0.05
    x1 = x1 * (1 - k)
    y1 = y1 * (1 - k)
    x2 = x2 * (1 + k)
    y2 = y2 * (1 + k)

    # get the image with the correct channels - gs,d,a
    # image = image[:, [0, 1, 2, 3], :, :]
    # # combine the rgb to grayscale by the formula
    # rgb = image[:, [0, 1, 2], :, :]
    # gs = torch.sum(rgb * torch.Tensor([0.2989, 0.5870, 0.1140]).unsqueeze(1).unsqueeze(1).unsqueeze(1), dim=1)
    # gs = gs.unsqueeze(1)
    # image = torch.cat((gs, image[:, 3:, :, :]), dim=1)

    # threshold the mask
    threshold = 0.5
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0

    cut = torch.cat(
        (image[:, 0:1, :, :] * 0.2989 + image[:, 1:2, :, :] * 0.5870 + image[:, 2:3, :, :] * 0.1140,
         image[:, 3:4, :, :]), dim=1)

    masked_image = cut * torch.Tensor(mask).unsqueeze(1)

    # center crop the image to the bbox
    masked_image_cropped = masked_image[:, :, int(y1):int(y2), int(x1):int(x2)]
    print((x1, y1), x2 - x1, y2 - y1)
    # convert to numpy
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='g', linewidth=3)
    plt.imshow(image[0, :3, :, :].permute(1, 2, 0).detach().numpy())
    plt.imshow(mask.squeeze(0), alpha=0.5)
    plt.gca().add_patch(rect)
    plt.scatter(centroid_x, centroid_y, c='r', s=10)
    plt.title("MRCNN_Results + Computed Centroid")
    plt.show()

    return masked_image_cropped, (centroid_x, centroid_y, depth_value), (x1, y1, x2, y2)


def vis_mask(rcimags, best, XY):
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(rcimags[0][:3, :, :].permute(1, 2, 0).detach().numpy())
    # best is just 1 mask in a dict
    mask = best[0]['masks'].detach().numpy()
    plt.imshow(mask[0], alpha=0.5, cmap='jet')
    plt.scatter(XY[0], XY[1], c='r', s=10)
    plt.title("MRCNN_Results + Computed Centroid")
    plt.show()


def viz_dir(hat_W, img, XYZ):
    from itertools import permutations
    hat_W = torch.stack([hat_W, hat_W], dim=0)
    magnitude = torch.sqrt(torch.sum(hat_W ** 2, dim=1)).view(-1, 1)

    hat_W = hat_W / magnitude
    hat_w = hat_W.cpu()
    img = img.cpu()
    XYZ = XYZ.cpu()

    hat_w = hat_w.detach().numpy()[0, :]
    # add axis with np.newaxis
    # stack them up
    hat_w = hat_w[np.newaxis, :]
    Img = img.permute(0, 2, 3, 1).detach().numpy()
    XYZ = XYZ.detach().numpy()

    Img = Img[:, :, :, :3]

    for img_index in range(hat_w.shape[0]):
        cur_img = Img[0]
        # cur_img = np.transpose(cur_img, (1, 2, 0))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cur_img[:, :, 0], cmap='gray')
        ax.set_title('Image with hat_w')

        start = np.array([cur_img.shape[1] // 2, cur_img.shape[0] // 2])

        scaling_factor = 100
        end_hat_w = start + scaling_factor * np.array([hat_w[img_index][0], hat_w[img_index][1]])

        # Create a set of points along the line for hat_w and gt_w
        num_points = 100
        points_hat_w = np.linspace(start, end_hat_w, num_points)

        # Plot hat_w line in green
        for i in range(num_points - 1):
            ax.plot(points_hat_w[i:i + 2, 0], points_hat_w[i:i + 2, 1], '-', color='g', alpha=0.5)

        plt.show()


def pnp_test(rgb, xyz, cut):
    # Set up the camera intrinsic parameters
    xyz = xyz[:, :, int(cut[1]):int(cut[3]), int(cut[0]):int(cut[2])]
    intrinsics = {
        'fx': 1181.077335,
        'fy': 1181.077335,
        'cx': 516.0,
        'cy': 386.0
    }
    camera_matrix = np.array([[intrinsics['fx'], 0, intrinsics['cx']],
                              [0, intrinsics['fy'], intrinsics['cy']],
                              [0, 0, 1]])
    dist_coeffs = np.zeros((4, 1))
    # Find the chessboard corners
    # Distortion coefficients (you should use the values from your specific camera)
    dist_coeffs = np.zeros(4, dtype=np.float32)

    def project_points(xyz, camera_matrix):
        # Project 3D points to 2D image plane
        projected_points, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), camera_matrix, np.zeros(4))
        return projected_points.reshape(-1, 2)

    # Estimate the pose using PnP
    xyz_reshaped = xyz.reshape(-1, 3)
    xyz_reshaped = xyz_reshaped.detach().numpy()
    points = project_points(xyz_reshaped, camera_matrix)

    # to numpy

    _, rvec, tvec = cv2.solvePnP(xyz_reshaped, points, camera_matrix, dist_coeffs)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    print("Rotation matrix: ", rotation_matrix)


import pathlib

if __name__ == '__main__':
    rcmodel, tmmodel, rcdata, device = prepare_inference()
    t = time.time()
    # get data from dataloader for maskrcnn
    rcdataiter = iter(rcdata)
    rcimages, rctargets = next(rcdataiter)
    rcimages, rctargets = next(rcdataiter)
    # rcimages, rctargets = next(rcdataiter)
    ptc = rcimages[:, [3, 4, 5], :, :]
    rcimages = rcimages[:, [0, 1, 2, 5, 6], :, :]
    rcimages = rcimages.to(device)
    print(rctargets[0]['tm'])

    best = infer_mrcnn(rcmodel, rcimages)

    masked_image_cropped, XYZ, world_coords = translation_layer(best, rcimages)

    # plt.imshow(masked_image_cropped[0, :3, :, :].permute(1, 2, 0).detach().numpy())
    # plt.show()

    pnp_test(masked_image_cropped.clone(), ptc, world_coords)
    masked_image_cropped = torch.nn.functional.avg_pool2d(masked_image_cropped, 3,
                                                          stride=1, padding=1)

    # vis_mask(rcimages, best, XYZ)

    # pass the data through the pose estimation network
    XYZ = torch.Tensor(XYZ).unsqueeze(0).to(device)

    # stackedimg = torch.cat(([masked_image_cropped] * 4), dim=0)
    # stackedxyz = torch.cat(([XYZ] * 4), dim=0)

    # tmoutputs = tmmodel(stackedimg, stackedxyz)
    tmoutputs = tmmodel(masked_image_cropped, XYZ)

    # get the predicted pose
    print("Predicted pose: ", tmoutputs[0].detach().numpy())

    # visualize the predicted pose
    viz_dir(tmoutputs[0], masked_image_cropped, XYZ)
    print("Time taken: ", time.time() - t)
