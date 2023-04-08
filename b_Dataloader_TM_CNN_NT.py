import os
import pathlib

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
from matplotlib import patches
from torch.utils.data import DataLoader, Subset, random_split
from scipy.spatial.transform import Rotation


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

        image_scaled = scale(image)
        # so the image is in the range of 0 to 1 not 0.5 to 1 as it is now
        # image_scaled[5] = (image_scaled[5] - 0.5) * 2
        # try:
        #     assert image_scaled[5].min() >= 0 and image_scaled[5].max() <= 1, "Image is not in the range of 0 to 1"
        # except AssertionError as e:
        #     print("Image min: ", image_scaled[5].min(), " max: ", image_scaled[5].max())
        #     raise e

        return image_scaled

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
        segmentation = eval(inst['segmentation'])

        decoded_mask = coco_mask.decode(segmentation)
        inst_mask = torch.from_numpy(decoded_mask).bool()

        inst_masks.append(inst_mask)
        inst_labels.append(inst['id'])

        # debug
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
    try:
        inst_masks = torch.nn.utils.rnn.pad_sequence(inst_masks, batch_first=True, padding_value=0)
    except RuntimeError as e:
        print("Error: ", e)
        print("inst_masks: ", inst_masks)
        raise e

    inst_masks = inst_masks.type(torch.bool)  # binary masks

    return inst_masks, inst_labels


def prepare_transforms(target):
    # Extract transformation matrices from target
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

    # now, the thinking goes like this:
    # Since the matrices are aligned on the axis with the infinite rotational symetry (Z axis)
    # We know that the rotation in the matrix will be aligned with the rotational symetry axis
    # So we can turn this problem into predicting the changed Z axis, in effect the change in coordinate system
    # This should basically dissregard the rotational symetry therefore symplifying the problem drastically
    # this line can be defined as point slope form with xyz being predicted / calculated from the mask
    # and the point being the center of mass of the mask
    # and slopes being predicted by the network

    return inst_rotations, inst_translations


def prepare_targets(targets) -> list:
    prepared_targets = []
    for target in targets:
        target = target[2:]
        if len(target) == 0:
            prepared_targets.append(None)
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

        bbox = [[inst['bbox'][0], inst['bbox'][1], inst['bbox'][0] + inst['bbox'][2],
                 inst['bbox'][1] + inst['bbox'][3]] for inst in target]
        # bbox = torch.cat(bbox, dim=0)
        #
        # bbox = bbox.type(torch.float32)
        prepared_target['box'] = bbox

        prepared_targets.append(prepared_target)

    # List of dictionaries - one dictionary per image
    return prepared_targets


def prepare_batch(batch):
    images, targets = zip(*batch)

    # Stack images on first dimension to get a tensor of shape (batch_size, C, H, W)
    batched_images = torch.stack(images, dim=0).permute(0, 3, 1, 2)
    del images

    # Prepare targets
    prepared_targets = prepare_targets(targets)
    tocut = []
    for i in range(len(prepared_targets)):
        tocut.append(False if prepared_targets[i] is None else True)

    batched_images = batched_images[tocut]

    prepared_targets = [target for target in prepared_targets if target is not None]

    return batched_images, prepared_targets


def collate_TM_GT(batch, channels=None):
    """Custom collate function to prepare data for the model"""
    images, targets = prepare_batch(batch)
    if not targets:
        return None, None, None, None, None
    if channels:
        images = images[:, channels, :, :]
    moves = [target['move'] for target in targets]
    # Get the Z translation for prediction as XY can be predicted from the mask
    zs = [move[:, 2] for move in moves]
    # zs = torch.stack(zs, dim=0)
    # XY seems to be usable from the PREDICTED mask because it has scoring, from the GT mask its bit more difficult
    # lets just use the TM gt for now
    xys = [move[:, :2] for move in moves]

    # Get the masks
    masks = [target['masks'] for target in targets]
    # Pad masks to all be the same size, needed for torch to stack them

    # Get the names
    names = [target['names'] for target in targets]
    # get rotation matrices
    rots = [target['rot'] for target in targets]

    # Extract the Z vector
    z_vecs = [rot[:, :, 2] for rot in rots]

    # change up the XY image to have the standard top left corner as origin
    # instead of the center of the image
    # this is needed for the crop to work correctly

    boxes = [target['box'] for target in targets]

    images_stacked_masked = []
    zs_stacked = []
    z_vecs_stacked = []
    XYZs_stacked = []

    # flatten and get the max area box
    all_boxes = []
    for box in boxes:
        all_boxes.extend(box)
    boxes = np.array(all_boxes)
    box_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    # get the coordinates of the box with the max area
    max_bbox = boxes[np.argmax(box_areas)]

    # then add ~10% to each side correctly to expand the bbox
    max_bbox_exp = [max_bbox[0] * 0.95, max_bbox[1] * 0.95, max_bbox[2] * 1.05, max_bbox[3] * 1.05]
    # then convert to int
    max_bbox_exp = [int(x) for x in max_bbox_exp]

    for idx, (image, mask, xy, z, z_vec, name, bbox) in enumerate(
            zip(images, masks, xys, zs, z_vecs, names, boxes)):
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

        # stack the image to match number of masks
        image = torch.stack([image for i in range(len(mask))])
        image_masks = [img * msk for img, msk in zip(image, mask)]
        # center crop the masked image according to the xys as the center
        image_masks_cropped = []
        XYZs = []
        w, h = max_bbox_exp[2] - max_bbox_exp[0], max_bbox_exp[3] - max_bbox_exp[1]
        w, h = w // 2, h // 2
        for j in range(len(image_masks)):
            img = image_masks[j]
            xyt = xy[j]
            zt = z[j].item()
            world_coords = (xyt[0], xyt[1], zt)
            world_coords = list(map(float, world_coords))
            # convert to image coordinates
            X, Y = world_to_image_coords(world_coords, intrinsics)
            XYZ = np.array([X, Y, zt])
            XYZs.append(XYZ)
            # compute the crop coordinates
            x1, y1, x2, y2 = int(X - w), int(Y - h), int(X + w), int(Y + h)

            # Calculate the padding required for each side
            pad_top = max(-y1, 0)
            pad_bottom = max(y2 - img.shape[1], 0)
            pad_left = max(-x1, 0)
            pad_right = max(x2 - img.shape[2], 0)

            # Pad the image only if required
            if pad_top or pad_bottom or pad_left or pad_right:
                img = np.pad(img, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")

            # Clip the coordinates to be within the image boundaries
            x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], a_min=0, a_max=None)

            # Crop the image using the clipped and padded coordinates
            img = img[:, y1:y2, x1:x2]

            image_masks_cropped.append(img if type(img) == torch.Tensor else torch.from_numpy(img))

            # debug viz
            # from matplotlib.colors import Normalize
            # from matplotlib.cm import ScalarMappable
            #
            # img = np.transpose(img, (1, 2, 0))
            # img = img[:, :, 0]
            #
            # fig, ax = plt.subplots(figsize=(10, 10))
            # ax.imshow(img)
            #
            # start = np.array([img.shape[1] // 2, img.shape[0] // 2])
            #
            # scaling_factor = 100
            # end = start + scaling_factor * np.array([z_vec[j][0], z_vec[j][1]])
            #
            # # Create a set of points along the line
            # num_points = 100
            # points = np.linspace(start, end, num_points)
            #
            # # Normalize the z-values of the points along the line
            # z_values = np.linspace(start[1] + scaling_factor * z_vec[j][2], end[1], num_points)
            # norm = Normalize(vmin=z_values.min(), vmax=z_values.max())
            # colors = plt.cm.viridis(norm(z_values))
            #
            # # Plot the line with color based on z-value (height)
            # for i in range(num_points - 1):
            #     ax.plot(points[i:i + 2, 0], points[i:i + 2, 1], '-', color=colors[i], alpha=0.8)
            #
            # # Add a colorbar
            # sm = ScalarMappable(cmap='viridis', norm=norm)
            # sm.set_array([])
            # plt.colorbar(sm, ax=ax, label='Height')
            #
            # plt.show()
            # print("z_vec", z_vec[j])


        # stack the images
        #image_masks_cropped = [torch.from_numpy(img) for img in image_masks_cropped]

        image_masks_stacked = torch.stack(image_masks_cropped)

        z = z.unsqueeze(1)
        zs_stacked.append(z)

        # z_vec = z_vec.unsqueeze(0)
        # concat 0 to the last position of the z_vec
        z_vecs_stacked.append(z_vec)

        # add the images to the list
        images_stacked_masked.append(image_masks_stacked)

        XYZs = np.array(XYZs)
        XYZs_stacked.append(XYZs)

    # stack the images
    images_stacked_masked = torch.cat(images_stacked_masked, dim=0)
    # stack the zs
    zs_stacked = torch.cat(zs_stacked, dim=0)
    # stack the z_vecs
    z_vecs_stacked = torch.cat(z_vecs_stacked, dim=0)
    # flatten the names
    names = [item for sublist in names for item in sublist]
    # stack the XYZs to tensor
    XYZs_stacked = np.concatenate(XYZs_stacked, axis=0)
    XYZs_stacked = torch.from_numpy(XYZs_stacked)
    images_stacked_masked, zs_stacked, z_vecs_stacked, names, XYZs_stacked = permute_microbatch(images_stacked_masked,
                                                                                                zs_stacked,
                                                                                                z_vecs_stacked,
                                                                                                names, XYZs_stacked)
    magnitude = torch.sqrt(torch.sum(z_vecs_stacked ** 2, dim=1)).view(-1, 1)

    # Normalize the z_vec_batch tensor
    z_vecs_stacked = z_vecs_stacked / magnitude

    # cut = 2
    # if zs_stacked.shape[0] > cut:
    #     images_stacked_masked = images_stacked_masked[:cut]
    #     zs_stacked = zs_stacked[:cut]
    #     z_vecs_stacked, names = z_vecs_stacked[:cut], names[:cut]
    #     XYZs_stacked = XYZs_stacked[:cut]

    # append 0 to the last position of the z_vecs_stacked to get shape (batch_size, 4)
    #z_vecs_stacked = torch.cat((z_vecs_stacked, torch.zeros(z_vecs_stacked.shape[0], 1)), dim=1)


    stack = (images_stacked_masked, zs_stacked, z_vecs_stacked, names, XYZs_stacked.type(torch.float32))
    return stack


def permute_microbatch(imgs, zs, z_vecs, names, XYZs):
    # say this is for better generalization
    # generate permutation
    perm = torch.randperm(imgs.shape[0])
    # permute the images
    imgs = imgs[perm]
    # permute the zs
    zs = zs[perm]
    # permute the z_vecs
    z_vecs = z_vecs[perm]
    # permute the names
    names = [names[i] for i in perm]
    # permute the XYZs
    XYZs = XYZs[perm]

    return imgs, zs, z_vecs, names, XYZs


class CollateWrapper:
    # Lambas are a nono in multiprocessing
    # need to be able to pickle the collate function to use it in multiprocessing, can't pickle lambdas
    def __init__(self, channels):
        self.collate_fn = collate_TM_GT
        self.channels = channels

    def __call__(self, batch):
        return self.collate_fn(batch, self.channels)


def createDataLoader(path, batchsize=1, shuffle=True, num_workers=4, channels: list = None, split=0.9):
    ano_path = (path.joinpath('annotations', "merged_coco.json"))

    collate = CollateWrapper(channels)

    # Load the COCO dataset
    dataset = CustomCocoDetection(root=str(path), annFile=str(ano_path))

    # subset the dataset
    #dataset = Subset(dataset, range(0, 200))

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
