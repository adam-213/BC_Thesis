import numpy as np


def D2FP16(depth):
    # max continuous values for fp16 is 4097 somehow, oh righ 4096 + a zero
    # so I need to scale the depth to -2048 to 2048
    # gives x16 precision compared to uint8
    # and therfore bigger resolution than scanner is capable of
    # max camera distance from the bin is 990mm, with scanner having 0.5mm precision
    # min camera distance from the full bin without overflow is say 550mm
    # I need 995-586 * 2 = 409mm of range but * 10 would be better and probably necessary numerically
    # I need to scale the depth to 0 - 4090 or -2045 to 2045 which is just within the range of fp16
    # giving me 0.1mm precision, plenty for current gen scanners

    # get the midpoints of the depth values - analytically empirically it doesn't make sense
    mid = (995 + 586) // 2  # floor division
    # only 3rd channel is depth, the other two are X and Y which can be extrapolated from pixels kinda
    # in a real scan you won't have this data anyway
    depth = depth[:, :, 2]
    # import matplotlib.pyplot as plt
    # depth[depth > 500] = -500
    # plt.hist(depth) # LIAR
    # plt.show()
    # scale the depth to -204.5 to 204.5
    depth = depth - mid
    # scale the depth to -2045 to 2045 == -2048 to 2048
    depth = depth * 10
    # convert to fp16
    depth = np.float16(depth)

    return depth


def RGB_PNG2FP16(image):
    # convert to fp16
    image = np.float16(image)

    return image


def RGB_EXR2FP16(image):
    # in fp32 0 - 1 scale to fp16 -2048 to 2048
    image = image * 4096 - 2048
    # convert to fp16
    image = np.float16(image)

    return image


def I2FP16(intensities):
    # in fp32 0 - 1 scale to fp16 -2048 to 2048
    image = intensities * 4096 - 2048
    # convert to fp16
    image = np.float16(image)

    return image


def N2FP16(normals):
    # in fp32 0 - 1 scale to fp16 -2048 to 2048
    image = normals * 4096 - 2048
    # convert to fp16
    image = np.float16(image)

    return image


def merge(*args):
    # handled in the colate function
    pass


def save_NPZ(arr_dict, save_path, index):
    # magic
    np.savez_compressed(str(save_path.joinpath(f'{index}.npz')), **arr_dict)
