import pathlib
import numpy as np
import os
import random as r

# enable cv to read exr files (It is critical this is done before cv is imported)
# It is disabled by default because of questionable security
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv


class Dataset:
    def __init__(self):
        # store all scan *paths* in memory as well (should be fine)
        self.scans = []

    def load_scan_paths(self):
        """Load all scan paths into memory"""
        path = pathlib.Path(__file__).parent.parent.absolute()
        data_path = path.joinpath('DataSets\\EXR\\')
        for folder in data_path.iterdir():
            if "T" in folder.name:
                continue
            scene_path = folder.joinpath('scene_paths.txt')
            with open(scene_path, 'r') as f:
                data = f.read()
                data = data.split('\n')
                for scan in data:
                    if scan:
                        scan_path = folder.joinpath(scan)
                        self.scans.append(scan_path)

    def paths(self, scene_path):
        scene_path = str(scene_path)
        path = \
            {
                "bin_transform": scene_path + "_bin_transform.txt",
                "corner_points": scene_path + "_corner_points.txt",
                "labels": scene_path + "_labels.png",
                "labels_info": scene_path + "_labels_info.txt",
                "part_transforms": scene_path + "_part_transforms.txt",
                "positions": scene_path + "_positions.exr",
            }
        return path

    def scan_load(self, scene_path):
        """Read single scan
        scene_path: $Bakalarka$\\DataSets\\EXR\\DS_XX_YY\\captures\\scan_xxx
        where XX is the dataset number and YY is the part number
        DS_XX_YY_M will be reserved for mixed bins if I ever figure out if that is possible
        """
        relative_path = self.paths(scene_path)
        bin_transform = load_bin_transform(relative_path["bin_transform"])
        corner_points = load_corner_points(relative_path["corner_points"])
        labels = load_labels(relative_path["labels"])
        labels_info = load_labels_info(relative_path["labels_info"])
        part_transforms = load_part_transforms(relative_path["part_transforms"])
        positions = load_exr_positons(relative_path["positions"])

        return tuple([bin_transform, corner_points, labels, labels_info, part_transforms, positions])

    def __getitem__(self, index):
        scan_path = self.scans[index]
        scan = self.scan_load(scan_path)
        #transformed_scan = self.transform(scan)
        return scan

    def __len__(self):
        return len(self.scans)


def load_bin_transform(path):
    """TODO Documentation Required"""
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


def load_corner_points(path):
    """TODO Documentation Required"""
    # read the corner points file
    with open(path, 'r') as f:
        lines = f.read()
    lines = lines.split("\n")
    corner_points = []
    for line in lines:
        if line:
            line = line.split(",")
            line = [float(i) for i in line]
            corner_points.append(line)
    # Returing the corner points, as a numpy array of float32 in original shape and order
    corner_points = np.array(corner_points).astype(np.float32)
    return corner_points


def load_labels(path):
    """TODO Documentation Required - stored in png because IDK"""
    # Binary file
    # read the labels file
    img = cv.imread(str(path), cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # convert to float32
    img = img.astype(np.float32)
    # get the shape of the image
    shape = img.shape
    # reshape the image to a 2d array
    img = img.reshape(shape[0] * shape[1], shape[2])
    # return the image
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
    """Position file is encoded as ZXY, with Z=0 being the camera position, XY = 0 being the center of the bin"""
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


if __name__ == "__main__":
    # Test the dataset
    import time
    from tqdm import tqdm
    dataset = Dataset()
    dataset.load_scan_paths()

    times = []
    for i in tqdm(range(len(dataset) // 8)):
        t = time.time()
        dataset[i]
        times.append(time.time() - t)

    print("Average time: ", sum(times) / len(times))
    # median
    times.sort()
    print("Median time: ", times[len(times) // 2])

    # modus
    print("Modus time: ", max(set(times), key=times.count))