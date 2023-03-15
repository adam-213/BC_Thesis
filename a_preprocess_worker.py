# This file converts raw scan data into usable data for the network
# For now it is run manually, but it can be automated in the future
# However there might be rather high processing costs for that

import multiprocessing as mp
# core
import pathlib

import pycocotools.coco as coco
# installed
from tqdm import tqdm

# local
from a_preprocess_IO_Functions import *
from a_preprocess_coco_creator import *
from a_preprocess_utils import *


class UniqueList(list):
    def append(self, object):
        if object not in self:
            super(UniqueList, self).append(object)

    def extend(self, iterable):
        for item in iterable:
            self.append(item)

    def where(self, value):
        for i, item in enumerate(self):
            if item == value:
                return i
        return None

    def sortself(self):
        self.sort()


class Preprocessor:

    def __init__(self):
        self.path = pathlib.Path(__file__).parent.absolute()
        self.data_path = self.path.joinpath('RawDS')
        self.coco_path = self.path.joinpath('COCO_TEST')
        self.save_path = self.coco_path.joinpath('train')

        if not self.save_path.exists():
            self.save_path.mkdir()

        if not self.coco_path.exists():
            self.coco_path.mkdir()

        if not self.coco_path.joinpath('annotations').exists():
            self.coco_path.joinpath('annotations').mkdir()

        self.scans = []
        self.api = coco.COCO()
        self.categories = UniqueList()

    def paths(self, scene_path):
        scene_path = str(scene_path)
        path = \
            {
                "bin_transform": scene_path + "_bin_transform.txt",
                "labels": scene_path + "_labels.png",
                "labels_info": scene_path + "_labels_info.txt",
                "part_transforms": scene_path + "_part_transforms.txt",
                "positions": scene_path + "_positions.exr",
                "intensities": scene_path + "_intensities.exr",
                "colors": scene_path + "_colors.exr",
                "normals": scene_path + "_normals.exr"
            }
        return path

    def load_scan_paths(self):
        """Load all scan paths into memory"""
        for folder in self.data_path.iterdir():
            scene_path = folder.joinpath('scene_paths.txt')
            with open(scene_path, 'r') as f:
                data = f.read()
                data = data.split('\n')
                for scan in data:
                    if scan:
                        scan_path = folder.joinpath(scan)
                        self.scans.append(scan_path)

    def scan_load(self, scene_path):
        """Read single scan
        scene_path: $Bakalarka$\\DataSets\\EXR\\DS_XX_YY\\captures\\scan_xxx
        where XX is the dataset number and YY is the part number
        DS_XX_YY_M will be reserved for mixed bins if I ever figure out if that is possible
        """
        relative_path = self.paths(scene_path)
        bin_transform = load_bin_transform(relative_path["bin_transform"])
        labels = load_labels(relative_path["labels"])
        # Doesn't realy matter because I'm working with one part type at a time - but it might be useful in the future
        labels_info = load_labels_info(relative_path["labels_info"])
        part_transforms = load_part_transforms(relative_path["part_transforms"])
        positions = load_exr_positons(relative_path["positions"])
        intensities = load_exr_intensities(relative_path["intensities"])
        rgb = load_exr_colors(relative_path["colors"])
        normals = load_exr_normals(relative_path["normals"])

        return [bin_transform, labels, labels_info, part_transforms, positions, intensities, rgb, normals]

    def process(self, scan_path, i, stls):
        """Process single scan"""
        scan = self.scan_load(scan_path)
        # print("Processing scan: ", scan_path)

        # create dict of all the wanted data in fp16
        # inputs = {}
        # inputs['rgb'] = RGB_EXR2FP16(scan[6])
        # inputs['depth'] = D2FP16(scan[4])
        # inputs['normals'] = N2FP16(scan[7])
        # inputs['intensities'] = I2FP16(scan[5])

        # create dict of all the wanted data in fp32
        inputs = {}
        inputs['rgb'] = scan[6]
        inputs['depth'] = scan[4]
        inputs['normals'] = scan[7]
        inputs['intensities'] = scan[5]

        # scale everything to 0-1
        for key, value in inputs.items():
            inputs[key] = scale(value)

        # get shapes for the coco json
        try:
            shapes = inputs['rgb'].shape[1:]
        except UnboundLocalError:
            shapes = inputs['depth'].shape

        # create the image part of the json - for this specific scan
        # will need to be merged with the rest of the json
        image = create_image_json(shapes, i)

        # save the image array to a npz file with the correct name
        save_NPZ(inputs, self.save_path, i)

        # compute mean and std for normalization for each channel in the scan
        mean, std = [], []
        for key, value in inputs.items():
            value = value.reshape(value.shape[0], value.shape[1], -1)
            for channel in range(value.shape[2]):
                mean.append(np.mean(value[:, :, channel]))
                std.append(np.std(value[:, :, channel]))

        tm = scan[3]
        labels = scan[1]
        labels_info = scan[2]

        # replace inf with 0
        mean = [0 if np.isinf(x) else x for x in mean]
        std = [0 if np.isinf(x) else x for x in std]

        stats = [mean, std]

        # create the annotations part of the json - for this specific scan
        annotations = create_annotations_json(labels, labels_info, tm, i, scan[0], self.categories, inputs, stats,
                                              stls)

        # json understands about 4 dtypes, so we need to convert the numpy dtypes to python dtypes
        result = prepare_data(image, annotations)

        return result, mean, std

    def worker(self, use_mp=False):
        """Process all scans"""
        # create categoies by iterating over all scans - but only on labels_info
        print('Creating categories')
        for scan_path in self.scans:
            info = load_labels_info(self.paths(scan_path)["labels_info"])
            for key, value in info.items():
                self.categories.append(key)

        self.categories.sort()
        print("Categories: ", self.categories)

        # load stls for the parts so we can compute occlusion
        stls = {}
        stls["part_thruster"] = "stl/part_thruster1.stl"
        stls["part_cogwheel"] = "stl/part_cogwheel1.stl"

        # start muliprocessing pool
        if use_mp:
            pool = mp.Pool(mp.cpu_count())
            # process all scans
            results = []
            for i, scan_path in enumerate(self.scans):
                results.append(pool.apply_async(self.process, args=(scan_path, i, stls)))
            pool.close()
            pool.join()
            results = [r.get() for r in results]
            results, means, stds = zip(*results)
        else:
            results = []
            # for i, scan_path in enumerate(tqdm(np.array(self.scans)[list(np.random.randint(0, len(self.scans), 25))])):
            start = 3200
            stop = 303
            for i, scan_path in enumerate(tqdm(self.scans[start:])):
                i = i + start
                if i == stop + start:
                    res, means, stds = zip(*results)
                    # combine the results into one json
                    print('Creating json')
                    print("categories: ", self.categories)
                    create_json(res, self.categories, self.coco_path, means, stds, f"_End_{start}_{stop+start}")
                    return
                results.append(self.process(scan_path, i, stls))
                if i % 50 == 0 and i != start:
                    res, means, stds = zip(*results)

                    # combine the results into one json
                    print('Creating json')
                    print("categories: ", self.categories)
                    create_json(res, self.categories, self.coco_path, means, stds, i)

        print('Creating json')
        print("categories: ", self.categories)
        create_json(results, self.categories, self.coco_path, means, stds, "all")


if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.load_scan_paths()
    preprocessor.worker(use_mp=False)
    print('Done')
