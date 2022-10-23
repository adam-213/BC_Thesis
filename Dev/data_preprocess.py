import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import numpy as np

def merge_positions_intensities(positions, intensities):
    """The XY coordinates from the positions file are not required therefore we can merge it into the intensities file
    which #TODO looks like Grayscale image of the bin with the objects in it"""



def load_positions(path):
    """TODO Documentation Required"""
    # read the exr file
    img = cv.imread(str(path), cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # convert to float32
    img = img.astype(np.float32)
    # get the shape of the image
    shape = img.shape
    # reshape the image to a 2d array
    img = img.reshape(shape[0] * shape[1], shape[2])
    # return the image
    return img

def load_intensities(path):
    """TODO Documentation Required"""
    # read the exr file
    img = cv.imread(str(path), cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # convert to float32
    img = img.astype(np.float32)
    # get the shape of the image
    shape = img.shape
    # reshape the image to a 2d array
    img = img.reshape(shape[0] * shape[1], shape[2])
    # return the image
    return img