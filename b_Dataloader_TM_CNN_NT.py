import os
import pathlib
import random
import random
import cv2
import numpy as np
import torch
import torchvision.transforms
import torchvision.transforms.functional
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
from matplotlib import patches
from torch.utils.data import DataLoader, Subset, random_split
from scipy.spatial.transform import Rotation
from torchvision.transforms import transforms as T
from PIL import Image
import torchvision.transforms.functional as F
import math
from copy import deepcopy


# Custom loader is needed to load RGB-A images - (4 channels) which in my case are RGB-D images
class CustomCocoDetection(CocoDetection):

    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile, transforms)
        # get image mean and std from the coco dataset, if not present, assume the default values (3 channel)
        image_stats = self.coco.dataset.get("image_stats", {})
        self.mean = np.array(image_stats.get("mean", [0.485, 0.456, 0.406]))
        self.std = np.array(image_stats.get("std", [0.229, 0.224, 0.225]))

    def _load_image(self, id: int) -> torch.Tensor:
        # overrride the _load_image method to load the images from the npz files in fp16
        # print("Loading image: ", id, "")
        path = self.coco.loadImgs(id)[0]["file_name"]
        # print("Path: ", path, " ")
        npz_file = np.load(os.path.join(self.root, path))
        # way to get the keys of the npz file, and load them as a list in the same order
        img_arrays = [npz_file[key] for key in npz_file.keys()]
        # dstack with numpy because pytorch refuses to stack different number of channels reasons
        image = np.dstack(img_arrays).astype(np.float32)
        # Channels are in the order of
        # R,G,B, X,Y,Z, NX,NY,NZ ,I
        image = torch.from_numpy(image).type(torch.float32)

        return image

    def _load_target(self, id: int):
        # override the _load_target becaiuse the original one is doing some weird stuff
        # no idea why it is doing that
        x = self.coco.imgToAnns[id]
        return x


def scale(image):
    # image appears to be -1 to 1
    # scale to 0 to 1
    image = (image + 1) / 2
    # print("Image min: ", image.min(), " max: ", image.max())
    return image


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


def prepare_masks(target):
    # Masks
    inst_masks = []
    inst_labels = []
    for inst in target:
        mask = inst['mask']
        size = inst['size']
        segmentation = {'counts': mask, 'size': size}

        decoded_mask = coco_mask.decode(segmentation)
        inst_mask = torch.from_numpy(decoded_mask).bool()

        inst_masks.append(inst_mask)
        inst_labels.append(inst['id'])

        # debug plot
        # plot inst mask and bbox
        # fig, ax = plt.subplots(1)
        # ax.imshow(inst_mask)
        # bbox = [inst['bbox'][0], inst['bbox'][1], inst['bbox'][0] + inst['bbox'][2],
        #         inst['bbox'][1] + inst['bbox'][3]]
        # ax.imshow(inst_mask)
        # ax.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
        #                                       linewidth=1, edgecolor='r', facecolor='none'))
        # plt.show()

    # Pad masks to all be the same size, needed for torch to stack them
    inst_masks = torch.nn.utils.rnn.pad_sequence(inst_masks, batch_first=True, padding_value=0)

    inst_masks = inst_masks.type(torch.bool)  # binary masks

    return inst_masks, inst_labels


def prepare_transforms(target):
    # Extract transformation matrices from target
    try:
        inst_transforms = [torch.tensor(inst['transform']) for inst in target]
        # turn them into np arrays, so we can reshape them correctly
        inst_transforms = [np.array(inst_transform) for inst_transform in inst_transforms]
        # Order F because the matrices are in column major order (mathematicians ...)
        inst_transforms = [i.reshape((4, 4), order='F') for i in inst_transforms]
        # Extract rotation and translation
        inst_rotations = [inst_transform[:3, :3] for inst_transform in inst_transforms]
        # Extract translation - not actualy needed as target will use analytical methods from the predicted masks + lookup of depth
        inst_translations = [inst_transform[:3, 3] for inst_transform in inst_transforms]
        # turn them into torch tensors
        inst_rotations = [torch.from_numpy(inst_rotation).type(torch.float32) for inst_rotation in inst_rotations]
        inst_translations = [torch.from_numpy(inst_translation).type(torch.float32) for inst_translation in
                             inst_translations]
        # stack them into a single tensor - (N, 3, 3) dtype: torch.float32, N: number of instances (variable)
        inst_rotations = torch.stack(inst_rotations, dim=0)
        # stack them into a single tensor - (N, 3) dtype: torch.float32, N: number of instances (variable)
        inst_translations = torch.stack(inst_translations, dim=0)
    except RuntimeError as e:
        print("Error: ", e)
        print("inst_transforms: ", inst_transforms)
        return None, None

    # now, the thinking goes like this:
    # Since the matrices are aligned on the axis with the infinite rotational symetry (Z axis)
    # We know that the rotation in the matrix will be aligned with the rotational symetry axis
    # So we can turn this problem into predicting the changed Z axis, in effect the change in coordinate system
    # This should basically dissregard the rotational symetry, therefore symplifying the problem drastically,
    # this line can be defined as point slope form with xyz being predicted / calculated from the mask
    # and the point being the center of mass of the mask
    # and slopes being predicted by the network

    return inst_rotations, inst_translations


def prepare_targets(targets) -> list:
    prepared_targets = []
    for target in targets:
        if len(target) == 0:
            continue
        prepared_target = {}
        # Masks and mask labels
        inst_masks, inst_labels = prepare_masks(target)
        # Transformations
        inst_rot, inst_move = prepare_transforms(target)

        # Store in dictionary
        prepared_target['masks'] = inst_masks.float()
        prepared_target['rot'] = inst_rot
        prepared_target['move'] = inst_move
        # labels need to be int64, such overkill
        prepared_target['labels'] = torch.tensor(inst_labels, dtype=torch.int64)
        prepared_target['names'] = np.array([np.array(inst['name']) for inst in target])
        prepared_target['centroid'] = np.array([np.array(inst['centroid']) for inst in target])

        bbox = [[inst['bbox'][0], inst['bbox'][1], inst['bbox'][0] + inst['bbox'][2],
                 inst['bbox'][1] + inst['bbox'][3]] for inst in target]
        # bbox = torch.cat(bbox, dim=0)
        #
        # bbox = bbox.type(torch.float32)
        prepared_target['box'] = bbox

        # filter out bins and background -
        # filter out upper names containing 'bin' or 'background' - filtered in data preparation - ie not needed
        #filt = prepared_target['labels'] > 2
        prepared_target['masks'] = prepared_target['masks']#[filt, :, :]
        prepared_target['rot'] = prepared_target['rot']#[filt, :, :]
        prepared_target['move'] = prepared_target['move']#[filt, :]
        prepared_target['labels'] = prepared_target['labels']#[filt]
        prepared_target['names'] = prepared_target['names']#[filt]
        prepared_target['box'] = np.array(prepared_target['box']).tolist()#[filt].tolist()
        prepared_target['centroid'] = np.array(prepared_target['centroid']).tolist()#[filt].tolist()
        if len(prepared_target['labels']) == 0:
            prepared_target = None

        prepared_targets.append(prepared_target)

    # List of dictionaries - one dictionary per image
    return prepared_targets


def prepare_batch(batch):
    # Filter out images with no target
    batch = [x for x in batch if len(x) == 2 and x[1] is not None]
    images, targets = zip(*batch)
    filtered_images, filtered_targets = [], []
    for image, target in zip(images, targets):
        if len(target) != 0:
            filtered_images.append(image)
            filtered_targets.append(target)

    images = filtered_images
    targets = filtered_targets
    if len(images) == 0:
        return None, None
    # Stack images on the first dimension to get a tensor of shape (batch_size, C, H, W)
    batched_images = torch.stack(images, dim=0).permute(0, 3, 1, 2)
    del images

    # Prepare targets
    prepared_targets = prepare_targets(targets)
    # cut out images with no targets
    tocut = []
    for i in range(len(prepared_targets)):
        tocut.append(False if prepared_targets[i] is None else True)

    batched_images = batched_images[tocut]

    prepared = []
    for target in prepared_targets:
        if target is not None:
            prepared.append(target)
    prepared_targets = prepared

    return batched_images, prepared_targets


def collate_first_stage(images, targets, channels, gray):
    if not targets:
        return None, None
    if channels:
        images = images[:, channels, :, :]
    # if gray and rgb in channels:
    if gray and min([i in channels for i in [0, 1, 2]]):
        # turn rgb into gray without touching everything else
        images = torch.cat(
            (images[:, 0:1, :, :] * 0.2989 + images[:, 1:2, :, :] * 0.5870 + images[:, 2:3, :, :] * 0.1140,
             images[:, 3:, :, :]), dim=1)
        # blur the image
        images = torch.cat((torch.nn.functional.avg_pool2d(images[:, 0:1, :, :], 3, stride=1, padding=1),
                            images[:, 1:, :, :]), dim=1)
        images = torch.nn.functional.avg_pool2d(images, 3, stride=1, padding=1)

    return images, targets


def collate_second_stage(images, targets):
    gt_translation_vector = [target['move'] for target in targets]
    # Get the Z translation for prediction as XY can be predicted from the mask
    gt_world_depth = [move[:, 2] for move in gt_translation_vector]
    gt_world_coords = [move[:, :2] for move in gt_translation_vector]
    # Get the masks
    gt_masks = [target['masks'] for target in targets]
    # Get the names
    gt_label_names = [target['names'] for target in targets]
    # get rotation matrices
    gt_rotation_matrices = [target['rot'] for target in targets]
    # Extract the Z vector
    gt_z_direction_vectors = [rot[:, :, 2] for rot in gt_rotation_matrices]
    # Get the bounding boxes
    gt_boxes = [target['box'] for target in targets]
    # Get the Centroids
    gt_centroids = [target['centroid'] for target in targets]
    return_list = []
    for i in range(len(targets)):
        target_dict = {"gt_masks": gt_masks[i], "gt_world_depth": gt_world_depth[i],
                       "gt_world_coords": gt_world_coords[i],
                       "gt_label_names": gt_label_names[i], "gt_z_direction_vectors": gt_z_direction_vectors[i],
                       "gt_boxes": gt_boxes[i], "gt_centroids": gt_centroids[i]}
        return_list.append(target_dict)

    return images, return_list


def collate_third_stage(images, targets, fixed_size, box_expantion_ratio):
    if fixed_size:
        # VIT needs a fixed ish size input, so just use that as the crop size
        return 224
    boxes = [target['gt_boxes'] for target in targets]
    # calculate the area of the boxes
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    # get the width and height of the max area box
    max_area_box = boxes[np.argmax(areas)]
    x1, y1, x2, y2 = max_area_box
    # expand the box by a factor of box_expantion_ratio
    x1 = x1 * (1 - box_expantion_ratio)
    y1 = y1 * (1 - box_expantion_ratio)
    x2 = x2 * (1 + box_expantion_ratio)
    y2 = y2 * (1 + box_expantion_ratio)
    # get the width and height of the expanded box
    width = x2 - x1
    height = y2 - y1
    # get the max of the width and height, doing it so that the crop is square
    max_dim = max(width, height)
    # get the crop size
    crop_size = int(max_dim)
    return crop_size


def crop_tensor_with_padding(img_tensor, x1, y1, x2, y2):
    channels, height, width = img_tensor.shape
    y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)

    crop_width = x2 - x1
    crop_height = y2 - y1

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)

    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant',
                                             value=0)

    cropped_tensor = img_tensor[:, y1:y2, x1:x2]
    return cropped_tensor


def collate_fourth_stage(images, targets, crop_size):
    cropped_images = []

    for image, target in zip(images, targets):
        # it's all the same image, but needs to be stacked as the masks are different
        images_stacked_masked = [image * mask for mask in target['gt_masks']]
        # find the coords for the crop of each mask
        crops = [(w - crop_size // 2, h - crop_size // 2, w + crop_size // 2, h + crop_size // 2) for w, h, z
                 in target['gt_centroids']]
        # crop the images
        images_stacked_masked_cropped = [crop_tensor_with_padding(image, x1, y1, x2, y2) for image, (x1, y1, x2, y2) in
                                         zip(images_stacked_masked, crops)]
        cropped_images.append(images_stacked_masked_cropped)

    # stack the images
    cropped_images = [torch.stack(image) for image in cropped_images]

    return cropped_images, targets


def collate_viz(images, targets):
    z_vec = targets['gt_z_direction_vectors']
    for j, img in enumerate(images):
        # debug viz
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, :3]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        start = np.array([img.shape[1] // 2, img.shape[0] // 2])
        ax.scatter(start[0], start[1], c='r', s=10)
        scaling_factor = 100
        end = start + scaling_factor * np.array([z_vec[j][0], z_vec[j][1]])

        # Create a set of points along the line
        num_points = 100
        points = np.linspace(start, end, num_points)

        # Normalize the z-values of the points along the line
        z_values = np.linspace(start[1] + scaling_factor * z_vec[j][2], end[1], num_points)
        norm = Normalize(vmin=z_values.min(), vmax=z_values.max())
        colors = plt.cm.viridis(norm(z_values))

        # Plot the line with color based on z-value (height)
        for i in range(num_points - 1):
            ax.plot(points[i:i + 2, 0], points[i:i + 2, 1], '-', color=colors[i], alpha=0.8)

        # Add a colorbar
        sm = ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Height')

        plt.show()
        print("z_vec", z_vec[j])
        break


def collate_fifth_stage(images, targets):
    # stack the Z direction vectors - this is the output of the model
    gt_z_dirs = [target['gt_z_direction_vectors'] for target in targets]
    gt_z_dirs = torch.cat(gt_z_dirs, dim=0)
    # stack the centroids to get a batch,3 sized tensor which can be passed to the model
    gt_centroids = [torch.Tensor(target['gt_centroids']) for target in targets]
    gt_centroids = torch.cat(gt_centroids, dim=0)

    # stack images
    images = torch.cat(images, dim=0)

    return images, gt_z_dirs, gt_centroids


def collate_permute_microbatch(images, gt_z_dirs, gt_centroids):
    perm = torch.randperm(images.shape[0])
    images = images[perm]
    gt_z_dirs = gt_z_dirs[perm]
    gt_centroids = gt_centroids[perm]
    return images, gt_z_dirs, gt_centroids


def collate_TM_GT(batch, channels=None, gray=True):
    """Custom collate function to prepare data for the model"""
    # get things into the solution
    images, targets = prepare_batch(batch)
    # make sure there is something to work with and select a subset of channels
    images, targets = collate_first_stage(images, targets, channels, gray)
    # if there is nothing to work with, return None
    if images is None or targets is None:
        return None, None, None
        # extract the targets into a usable data structure (dict)
    images, gt_targets = collate_second_stage(images, targets)
    # get the crop size, either fixed or based on the max area box
    crop_size = collate_third_stage(images, gt_targets, fixed_size=True, box_expantion_ratio=0.1)
    # crop the images
    cropped_images, targets = collate_fourth_stage(images, gt_targets, crop_size)
    # visualize the images
    # collate_viz(images, targets, z_vec, j)
    # prepare the actuall target for the model + prepare centroids for possible input augmentation
    images, gt_z_dirs, gt_centroids = collate_fifth_stage(cropped_images, targets)
    # permute the microbatch
    images, gt_z_dirs, gt_centroids = collate_permute_microbatch(images, gt_z_dirs, gt_centroids)

    return images, gt_z_dirs, gt_centroids


class CollateWrapper:
    # Lambas are a nono in multiprocessing
    # need to be able to pickle the collate function to use it in multiprocessing, can't pickle lambdas
    def __init__(self, channels, gray=False):
        self.collate_fn = collate_TM_GT
        self.channels = channels
        self.gray = gray

    def __call__(self, batch):
        return self.collate_fn(batch, self.channels, self.gray)


def createDataLoader(path, batchsize=1, shuffle=True, num_workers=4, channels: list = None, split=0.9, gray=False):
    ano_path = (path.joinpath('annotations', "merged_maskrcnn_centroid2.json"))
    # ano_path = (path.joinpath('annotations', "merged_coco.json"))

    collate = CollateWrapper(channels, gray=gray)

    # Load the COCO dataset
    dataset = CustomCocoDetection(root=str(path), annFile=str(ano_path))


    # compute class samples
    cattoimgs = dataset.coco.catToImgs
    cattoimgs = {k: len(v) for k, v in cattoimgs.items()}
    # look up the names
    cattoimgs = {dataset.coco.cats[k]['name']: v for k, v in cattoimgs.items()}

    # sort the dict
    cattoimgs = {k: v for k, v in sorted(cattoimgs.items(), key=lambda item: item[1])}
    # print it one by one in new line
    for k, v in cattoimgs.items():
        print(k, v)




    # Calculate the lengths of the train and validation sets
    train_len = int(len(dataset) * split)
    val_len = len(dataset) - train_len

    # Create the train and validation subsets
    train_set, val_set = random_split(dataset, [train_len, val_len])

    # Create the PyTorch dataloaders for train and validation sets
    train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers,
                                  collate_fn=collate)
    val_dataloader = DataLoader(val_set, batch_size=batchsize, shuffle=False, num_workers=num_workers,
                                collate_fn=collate)

    return train_dataloader, val_dataloader

# some comments from the old ish code , code that is not used anymore but might be useful in the future

# change up the XY image to have the standard top left corner as origin
# instead of the center of the image
# this is needed for the crop to work correctly
# for each mask there is a xy, for this xy i want to predict z
# I want to crop them to reasonable size but since they are irelular i need to pad them
# So I want to find the biggest bbox do 1.1x and then say that is the crop size
# image * mask = masked image
# center crop the masked image according to the xys as the center
# since I have already calculated the rectangle size I can just use that and they will be the same size
# then I can just stack them and feed them into the network with the zs as the target
# this is the first stage of the training

# second stage, take the same images and crop them to the same size as the first stage
# then I can just stack them and feed them into the network with the z_vecs as the target
# but this time using the forward_s2 method instead of forward_s1
# this is the second stage of the training

# determine if the weights should be freezed or not, args against is that the backbone will be changing
# therefore the Z head won't have the same wiegghts as it had when S1 traing was done
# therefore it might drift away from the correct solution that it found in S1

# potentialy i could freeze tha backbone and Z head and only train the W head however this will probably be worse
# in terms of accuracy since the backbone is the most important part of the network

# firs max the size of the bbox

# normalisation turned off on purpose for trying out the theory that it hurts the performance
# it should technically be normalised by default - not sure
# magnitude = torch.sqrt(torch.sum(z_vecs_stacked ** 2, dim=1)).view(-1, 1)
#
# # Normalize the z_vec_batch tensor
# z_vecs_stacked = z_vecs_stacked / magnitude

# cut = 30
# if zs_stacked.shape[0] > cut:
#     images_stacked_masked = images_stacked_masked[:cut]
#     zs_stacked = zs_stacked[:cut]
#     z_vecs_stacked, names = z_vecs_stacked[:cut], names[:cut]
#     XYZs_stacked = XYZs_stacked[:cut]

# append 0 to the last position of the z_vecs_stacked to get shape (batch_size, 4)
# z_vecs_stacked = torch.cat((z_vecs_stacked, torch.zeros(z_vecs_stacked.shape[0], 1)), dim=1)
# rotatae each emage in the batch by a random angle

# Create a list of rotated images - not a good idea
# rotated_images = []
# for i in range(images_stacked_masked.shape[0]):
#     # Extract the i-th image from the batch
#     image = images_stacked_masked[i, :, :, :]
#     angle = random.randint(0, 360)
#
#     # Apply a random rotation to the image
#     rotated_image = F.rotate(image, angle)
#
#     # Add the rotated image to the list
#     rotated_images.append(rotated_image)
#
# # Stack the rotated images into a new tensor
# images_stacked_rotated = torch.stack(rotated_images, dim=0)
