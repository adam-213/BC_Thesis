import time

import numpy

from b_DataLoader_RCNN import createDataLoader as rcdataloader

from d_TM_Eff_NT import PoseEstimationModel as peModel
from b_MaskRCNN import MaskRCNN as rcModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torchvision

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
    coco_path = base_path.joinpath('CCO_TE')
    channels = [0, 1, 2, 3, 4, 5, 9]
    chans_sel = [0, 1, 2, 5, 6]
    # create unshuffled dataloaders
    rcdata, val_dataloader, stats = rcdataloader(coco_path, 1, channels=channels, shuffle=True, num_workers=0,
                                                 anoname="merged.json")
    mean, std = stats
    mean, std = np.array(mean), np.array(std)
    # cut out the channels we don't need
    sel = np.array(channels)[chans_sel]
    mean, std = mean[sel], std[sel]

    rcmodel = rcModel(5, 12, mean, std)

    # create models
    tmmodel = peModel(3)

    # load weights
    tmmodel.load_state_dict(torch.load('Unscaled_80_stage2.pth')['model_state_dict'])
    rcmodel.load_state_dict(torch.load(base_path.joinpath("RCNN_Unscaled_34.pth"))['model_state_dict'])

    # set models to eval mode
    tmmodel.eval()
    rcmodel.eval()

    # set device to cpu
    device = torch.device('cpu')
    tmmodel.to(device)
    rcmodel.to(device)

    return rcmodel, tmmodel, rcdata, device


def infer_mrcnn(model, image):
    with torch.no_grad():
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


def geometric(ptc, mask, tensor=False):
    point_cloud = (ptc.squeeze(0) * torch.Tensor(mask)).reshape(3, -1)
    valid_points = point_cloud[:, torch.any(point_cloud != 0, axis=0)]

    if valid_points.size(1) != 0:
        geometric_centroid = torch.mean(valid_points, axis=1)
        if tensor:
            return geometric_centroid
        else:
            return geometric_centroid[0].item(), geometric_centroid[1].item(), geometric_centroid[2].item()
    else:
        if tensor:
            return torch.tensor([float('inf'), float('inf'), float('inf')])
        else:
            return float('inf'), float('inf'), float('inf')


def translation_layer(best, image, ptc):
    # Prepare data for pose estimation
    # get bbox
    bbox = best[0]['boxes'].detach().numpy()
    # get mask
    mask = best[0]['masks'].detach().numpy()

    # threshold the mask
    threshold = 0.8
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0

    # get centroid

    centroid = geometric(ptc, mask, tensor=False)
    centroid_x, centroid_y = world_to_image_coords((centroid[0], centroid[1],centroid[2]),
                                                   intrinsics)
    print(centroid)

    # calculate xy for cut
    cx, cy = centroid_x, centroid_y
    # calculate bbox
    size = 112
    x1, y1, x2, y2 = cx - size , cy - size , cx + size, cy + size

    depth_value = centroid[2]

    cut = torch.cat(
        (image[:, 0:1, :, :] * 0.2989 + image[:, 1:2, :, :] * 0.5870 + image[:, 2:3, :, :] * 0.1140,
         image[:, 3:5, :, :]), dim=1)

    # print(cut.shape)

    masked_image = cut * torch.Tensor(mask).unsqueeze(1)

    # center crop the image to the bbox
    masked_image_cropped = masked_image[:, :, int(y1):int(y2), int(x1):int(x2)]

    if depth_value == 0:
        notzero = masked_image_cropped[0, 1, :, :]
        depth_value = torch.mean(notzero[notzero != 0])
        print("Depth value is 0, using mean depth value instead", depth_value)

    # print((x1, y1), x2 - x1, y2 - y1)
    # convert to numpy
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='g', linewidth=3)
    plt.imshow(image[0, :3, :, :].permute(1, 2, 0).detach().numpy())
    plt.imshow(mask.squeeze(0), alpha=0.5)
    plt.gca().add_patch(rect)
    plt.scatter(centroid_x, centroid_y, c='r', s=10)
    plt.title(f"MRCNN_Results + Computed Centroid - {depth_value}")
    plt.show()

    return masked_image_cropped, (centroid_x, centroid_y, depth_value), centroid


def vis_mask(rcimags, best, XY):
    fig, ax = plt.subplots(1, 1, figsize=(25, 25))
    ax.imshow(rcimags[0][:3, :, :].permute(1, 2, 0).detach().numpy())
    # best is just 1 mask in a dict
    mask = best[0]['masks'].detach().numpy()
    plt.imshow(mask[0], alpha=0.5, cmap='jet')
    plt.scatter(XY[0], XY[1], c='r', s=10)
    plt.title("MRCNN_Results + Computed Centroid")
    plt.show()


def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def viz_dir(hat_W, img, XYZ, gt_zvec, loss):
    from itertools import permutations
    hat_W = torch.stack([hat_W, hat_W], dim=0)
    magnitude = torch.sqrt(torch.sum(hat_W ** 2, dim=1)).view(-1, 1)

    hat_W = hat_W / magnitude
    hat_w = hat_W.cpu()
    img = img.cpu()
    XYZ = XYZ.cpu()

    hat_w = hat_w.detach().numpy()[:1, :]
    gt_w = gt_zvec.detach().numpy()[:, np.newaxis]

    Img = img.permute(0, 2, 3, 1).detach().numpy()
    XYZ = XYZ.detach().numpy()

    Img = Img[:, :, :, :3]

    for img_index in range(hat_w.shape[0]):
        cur_img = Img[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(cur_img[:, :, 0], cmap='gray')
        ax1.set_title(f'Image with hat_w and gt_w - Loss: {loss}')

        start = np.array([cur_img.shape[1] // 2, cur_img.shape[0] // 2])

        scaling_factor = 100
        end_hat_w = start + scaling_factor * np.array([hat_w[img_index][0], hat_w[img_index][1]])
        end_gt_w = start + scaling_factor * np.array([gt_w[0][0], gt_w[1][0]])

        num_points = 100
        points_hat_w = np.linspace(start, end_hat_w, num_points)
        points_gt_w = np.linspace(start, end_gt_w, num_points)

        # Plot hat_w line in green
        for i in range(num_points - 1):
            ax1.plot(points_hat_w[i:i + 2, 0], points_hat_w[i:i + 2, 1], '-', color='g', alpha=0.5)

        # Plot gt_w line in red
        for i in range(num_points - 1):
            ax1.plot(points_gt_w[i:i + 2, 0], points_gt_w[i:i + 2, 1], '-', color='r', alpha=0.5)

        # 3D subplot with arcs
        ax2 = fig.add_subplot(122, projection='3d')

        # Draw octant separation planes with different colors and 0.2 alpha
        xx, yy = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2))
        zz = np.zeros_like(xx)
        ax2.plot_surface(xx, yy, zz, alpha=0.2, color='r')  # X-Y plane (red)
        ax2.plot_surface(xx, zz, yy, alpha=0.2, color='g')  # X-Z plane (green)
        ax2.plot_surface(zz, yy, xx, alpha=0.2, color='b')  # Y-Z plane (blue)

        # Plot the arc
        num_arc_points = 100
        arc_points = np.array([slerp(hat_w[img_index], gt_w[:, 0], t) for t in np.linspace(0, 1, num_arc_points)])
        ax2.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color='r', alpha=0.5)

        # Plot hat_w and gt_w vectors
        ax2.quiver(0, 0, 0, hat_w[img_index][0], hat_w[img_index][1], hat_w[img_index][2], color='g', alpha=0.8,
                   arrow_length_ratio=0.1)
        ax2.quiver(0, 0, 0, gt_w[0][0], gt_w[1][0], gt_w[2][0], color='r', alpha=0.8, arrow_length_ratio=0.1)
        from math import pi, cos
        A, B = hat_w[img_index], gt_w[:, 0]
        dot = np.dot(A, B)
        magnitude_A = np.linalg.norm(A)
        magnitude_B = np.linalg.norm(B)
        theta_degrees = np.arccos(dot / (magnitude_A * magnitude_B)) * 180 / np.pi
        ax2.set_title(f'3D plot of hat_w and gt_w - Loss: {loss} - Angle: {theta_degrees:.2f} degrees')

        # Set the limits and aspect ratio
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_zlim(-1, 1)
        ax2.set_box_aspect((1, 1, 1))

        plt.show()


import pathlib


def find_match(gt, best):
    # gt is a list of dicts
    # best is a list of dicts

    bbox = best[0]['boxes']
    mask = best[0]['masks']

    for box, tmask, tm in zip(gt[0]['boxes'], gt[0]['masks'], gt[0]['tm']):
        box = box.type(torch.int32)
        # box_iou = box_iou(box, bbox)

        intersection = torch.sum(mask * tmask).float()
        union = torch.sum(mask + tmask - mask * tmask).float()
        iou = intersection / union

        if iou > 0.75:
            return tm


def main():
    rcmodel, tmmodel, rcdata, device = prepare_inference()
    t = time.time()
    # get data from dataloader for maskrcnn
    rcdataiter = iter(rcdata)
    rcimages, rctargets = next(rcdataiter)
    # rcimages, rctargets = next(rcdataiter)
    # rcimages, rctargets = next(rcdataiter)
    ptc = rcimages[:, [3, 4, 5], :, :]
    rcimages = rcimages[:, [0, 1, 2, 5, 6], :, :]
    rcimages = rcimages.to(device)
    # print(rctargets[0]['tm'])

    best = infer_mrcnn(rcmodel, rcimages)
    try:
        gttm = find_match(rctargets, best)
    except:
        gttm = torch.from_numpy(np.Identity(4)).float()
        print("no match found")

    masked_image_cropped, XYZ, world_coords = translation_layer(best,
                                                                rcimages,
                                                                ptc)  # TODO add some sort of centroid heuristic

    # vis_mask(rcimages, best, XYZ)

    # pass the data through the pose estimation network
    XYZ = torch.Tensor(XYZ).unsqueeze(0).to(device)
    print("coords", XYZ)

    tmoutputs = tmmodel(masked_image_cropped, XYZ)
    try:
        gt_zvec = gttm[:3, 2]
    except:
        gt_zvec = torch.Tensor([0, 0, 1]).to(device)
        print("no match found")
    print("Ground truth pose: ", gt_zvec.detach().numpy())
    # print("Predicted pose: ", tmoutputs[:1].detach().numpy())
    loss = tmmodel.Wloss(tmoutputs[:1, :], gt_zvec.unsqueeze(0).to(device))

    # get the predicted pose
    print("Predicted pose: ", tmoutputs[:1].detach().numpy())

    # visualize the predicted pose
    viz_dir(tmoutputs[0], masked_image_cropped, XYZ, gt_zvec, loss)
    print("Time taken: ", time.time() - t)
    templabels = {10:"part_thruster_normalized_centered",4:"part_cogwheel_normalized_centered",9:"part_cogwheel_normalized_centered"}
    #label_name = rcdata.dataset.dataset.coco.cats[best[0]["labels"].item()]["name"]
    label_name = templabels[best[0]["labels"].item()]

    return (
        tmoutputs[:1].detach().numpy(),  # predicted zvec
        XYZ,  # centroid + looked up depth
        best[0]["masks"],  # MaskRCNN mask
        label_name,  # MaskRCNN label to get correct stl
        rcimages.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),  # MaskRCNN input image to get the depth map
        gttm,  # Ground truth pose,
        ptc,  # Point cloud,
        world_coords,  # World coordinates

    )


if __name__ == '__main__':
    main()
