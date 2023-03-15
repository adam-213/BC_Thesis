import math
import os
import random

import bpy
import mathutils
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation


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


def scale(arr):
    # scale to 0 - 1
    # arr = arr - np.min(arr)
    arr = arr / np.max(arr)

    return arr


# Probably not actually used, but I'm keeping it just in case
def apply_base_rotation(T, rot):
    """The default orientation of some objects is just super wierd, and totaly not intuitive. This function
    rotates the object to a more intuitive orientation."""

    # create a rotation object representing the desired orientation
    rot = Rotation.from_euler('xyz', rot, degrees=True)
    RT = rot.as_matrix()

    # 1x16 to 4x4
    T = T.reshape((4, 4))

    # combine the rotation matrix with the transformation matrix
    T[:3, :3] = RT @ T[:3, :3]

    return T


def render(path, T):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Load the STL file
    bpy.ops.import_mesh.stl(filepath=path)

    # Get a reference to the imported object
    obj = bpy.context.selected_objects[0]

    # Convert the transformation matrix to a list of python floats
    T = T.reshape((4, 4), order='F')

    # Apply the transformation matrix to the object
    obj.matrix_world @= mathutils.Matrix(T)

    material = bpy.data.materials.new(name='Material')
    obj.active_material = material

    # Set up the camera with the given parameters
    camera = bpy.data.cameras.new('Camera')
    camera_obj = bpy.data.objects.new('Camera', camera)
    bpy.context.scene.camera = camera_obj
    camera_obj.location = (0, 0, 0)
    camera_obj.rotation_euler = (math.radians(180), 0, 0)
    camera_obj.data.angle = math.radians(11.75)
    camera_obj.data.sensor_width = 47.50
    camera_obj.data.sensor_height = 36
    camera_obj.data.lens = 1181.0774 / camera_obj.data.sensor_height * 1.655
    camera_obj.data.shift_x = (516.0 - 1032 / 2) / camera_obj.data.sensor_height
    camera_obj.data.shift_y = (386.0 - 772 / 2) / camera_obj.data.sensor_width

    # Set the near and far clipping planes
    bpy.context.scene.camera.data.clip_start = 458.0
    bpy.context.scene.camera.data.clip_end = 1118.0

    bpy.context.scene.collection.objects.link(camera_obj)

    # Render the scene and get the rendered image
    # Set render settings
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.resolution_x = 1032
    bpy.context.scene.render.resolution_y = 772
    bpy.context.scene.render.film_transparent = True

    # Set EEVEE settings
    bpy.context.scene.eevee.use_bloom = False
    bpy.context.scene.eevee.use_ssr = False
    bpy.context.scene.eevee.use_soft_shadows = False
    bpy.context.scene.eevee.taa_render_samples = 8
    bpy.context.scene.eevee.taa_samples = 8
    # bpy.context.scene.eevee.motion_blur_samples = 1

    # Simplify the scene
    bpy.context.scene.render.use_simplify = True
    bpy.context.scene.render.simplify_subdivision_render = 1

    # save the image to the buffer
    file = f'R:\\dummy{random.randint(10000, 12312312312)}.png'
    bpy.context.scene.render.filepath = file
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(
        use_viewport=True,
        write_still=True,
        scene=bpy.context.scene.name,
    )
    # Load the rendered image from the memory buffer and get the pixel data
    image = np.array(Image.open(file))

    delete(bpy.context.scene)

    # remove the temp file
    os.remove(file)
    return image


def delete(scene):
    """Something from this helps with memory leaks, but I'm not sure what"""
    # Loop through all objects in the scene and delete them
    for obj in scene.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Delete all materials
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat, do_unlink=True)

    # Delete all textures
    for tex in bpy.data.textures:
        bpy.data.textures.remove(tex, do_unlink=True)

    # Delete all images
    for img in bpy.data.images:
        bpy.data.images.remove(img, do_unlink=True)

    # Delete all meshes
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)

    # Delete all armatures
    for arm in bpy.data.armatures:
        bpy.data.armatures.remove(arm, do_unlink=True)

    # Delete all actions
    for action in bpy.data.actions:
        bpy.data.actions.remove(action, do_unlink=True)

    # Delete all curves
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve, do_unlink=True)

    # Delete all fonts
    for font in bpy.data.fonts:
        bpy.data.fonts.remove(font, do_unlink=True)


def is_mask_good(mask, image, category, stls, T, occlusion=0.17):
    # i cant believe this works finally

    obj = render(stls[category], T)

    # threshold the object to a binary mask
    render_mask = obj[:, :, 3] > 0

    # get the overlap between the two masks
    overlap = np.logical_and(mask, render_mask)

    # compute number of segments of the mask with ndimage
    labeled_mask, num_features = ndimage.label(mask)

    # # ### DEBUGGING ###
    # # plot the masks for debugging
    # fig, ax = plt.subplots(figsize=(10, 10))
    # # draw mask in red channel
    # # draw render mask in green channel
    #
    # img = np.zeros((image["rgb"].shape[0], image["rgb"].shape[1], 3))
    # img[:, :, 0] = mask * 255
    # img[:, :, 1] = render_mask * 255
    #
    # ax.imshow(img)
    # ax.imshow(image["intensities"], alpha=0.5)
    # # add area of the mask
    # ax.text(0, 0, f"Mask Area: {np.sum(mask)}", color="red")
    # # add area of render mask
    # ax.text(0, 50, f"Render Mask Area: {np.sum(render_mask)}", color="red")
    # # add area of overlap
    # ax.text(0, 100, f"Overlap Area: {np.sum(overlap)}", color="red")
    # # add percentage of overlap
    # ax.text(0, 150, f"Overlap Percentage: {np.sum(overlap) / np.sum(render_mask)}", color="red")
    # # add percentage of occlusion
    # ax.text(0, 200, f"Occlusion Percentage: {occlusion}", color="red")
    # # add OK
    # ax.text(0, 250, f"OK: {np.sum(overlap) / np.sum(render_mask) > 1 - occlusion}", color="red")
    #
    # fig.savefig(f"E:\\zmasked{random.randint(1000, 100000)}.png")
    # #
    # # ### DEBUGGING ###

    if np.sum(overlap) / np.sum(render_mask) > 1 - occlusion and num_features == 1:
        return True
    return False
