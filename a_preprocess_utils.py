import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import numpy as np


def merge2RGBD(rgb, depth):
    # scale depth to 0-255
    # in such a way that i have at least 0.5 precision
    # i'm counting on the depth being max 8
    depth = depth[:, :, 2]
    depth = depth - 795  # -255 to 255
    depth = depth + 256  # 0 to 510
    depth = depth / 2  # 0 to 255
    assert np.all(np.sort(np.unique(depth))[1:]) >= 0
    np.clip(depth, 0, 255, out=depth)
    depth = np.uint8(depth)

    # rgb is in int8 convert to float16, so it can be merged with depth
    # rgb = np.float16(rgb)

    # Merge the rgb and depth into a single image
    rgbd = np.dstack((rgb, depth))
    rgbd = rgbd.astype(np.uint8)
    # print(rgbd.shape)
    # should be (h,w, 4)

    # TODO: downsample the image by 2, probably not needed its not that big
    # downsample the image by 2 #  this might be able to be done in the generator
    # I don't think this will work because labels are not downsampled
    # and i'm not sure how to downsample them
    # rgbd = cv.resize(rgbd, (0, 0), fx=0.5, fy=0.5)

    # This should return half size image with around 2mm accuracy in depth which should be enough
    # for the task at hand

    return rgbd


def save_rgbd(rgbd, save_path, i):
    # save as npz compressed
    # np.savez_compressed(str(save_path.joinpath(str(i) + '.npz')), rgbd)
    # save as png
    cv.imwrite(str(save_path.joinpath(str(i) + '.png')), rgbd)

#
