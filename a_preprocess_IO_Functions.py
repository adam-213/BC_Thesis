import os

# enable cv to read exr files (It is critical this is done before cv is imported)
# It is disabled by default because of questionable security

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import numpy as np
import pickle
import bz2


def load_bin_transform(path):
    """Transformation matrix for the bin"""
    # read the transform file
    with open(path, 'r') as f:
        # read the first line
        line = f.readline()
        # split the line into a list
        line = line.split()
        # convert the list to a numpy array
        line = np.array(line)
        # convert the array to a float32
        line = line.astype(np.float32)
        # return the array
        return line


def load_labels(path):
    """Each pixel in the image labeled with a number, stored in a png file"""
    # Binary file
    # read the labels file
    img = cv.imread(str(path), cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # only one channel is used
    img = img[:, :, 2]
    return img


def load_labels_info(path):
    """TODO Documentation Required"""
    with open(path, 'r') as f:
        lines = f.read()
    lines = lines.split("\n")
    labels_info = {}
    for line in lines:
        if line:
            line = line.split(" ")
            if line[1] in labels_info:
                labels_info[line[1]].append(int(line[0]))
            else:
                labels_info[line[1]] = [int(line[0])]

    return labels_info


def load_part_transforms(path):
    """TODO Documentation Required"""
    with open(path, 'r') as f:
        lines = f.read()
    lines = lines.split("\n")
    part_transforms = []
    for line in lines:
        if line:
            line = line.split(" ")
            line = [float(i) for i in line]
            part_transforms.append(line)
    # Returing the corner points, as a numpy array of float32 in original shape and order
    part_transforms = np.array(part_transforms).astype(np.float32)
    return part_transforms


def load_exr_positons(path):
    """Read the raw positions from the exr file, function only used for preprocessing"""
    # cv.COLORMAP_HSV and cv.IMREAD_ANYDEPTH are very necessary otherwise we are stuck with uint8
    img = cv.imread(str(path), cv.COLORMAP_HSV | cv.IMREAD_ANYDEPTH)

    return img


def load_exr_intensities(path):
    """TODO Documentation Required"""
    # read the exr file
    # cv.COLORMAP_HSV and cv.IMREAD_ANYDEPTH are very necessary otherwise we are stuck with uint8
    img = cv.imread(str(path), cv.COLORMAP_HSV | cv.IMREAD_ANYDEPTH)

    return img


def load_exr_colors(path):
    img = cv.imread(str(path), cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # convert to 0-255 range
    img = img * 255
    return img
