import bpy
import numpy as np
import mathutils, math
import random, os
import cv2

import os
import sys
from contextlib import contextmanager


@contextmanager
def stdout_redirected(to=os.devnull):
    # a workaround for the fact that blender is more verbose than your mom when they meet an old friend
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


class STL_renderer:
    def __init__(self, INTRINSICS, width, height):
        self.INTRINSICS = INTRINSICS
        self.width = width
        self.height = height
        self.obj = None
        self.output_path = "R:\\temp_data.png"
        self.iter = 0

        self.setup_scene()

    def setup_scene(self):
        self.delete(bpy.context.scene)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.setup_camera()
        self.create_point_light((0, 0, 0), energy=20000000)

        # Set render settings
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.resolution_x = self.width
        bpy.context.scene.render.resolution_y = self.height

        # Set EEVEE settings
        bpy.context.scene.eevee.use_bloom = False
        bpy.context.scene.eevee.use_ssr = False
        bpy.context.scene.eevee.use_soft_shadows = False
        bpy.context.scene.eevee.taa_render_samples = 4
        bpy.context.scene.eevee.taa_samples = 2

        # Simplify the scene
        bpy.context.scene.render.use_simplify = True
        bpy.context.scene.render.simplify_subdivision_render = 1
        # bpy.context.scene.render.image_settings.file_format = 'PNG'

        # setup the output as BW
        bpy.context.scene.render.image_settings.color_mode = 'BW'

    def setup_camera(self):
        # Create a new camera in Blender
        camera = bpy.data.cameras.new('Camera')
        camera_obj = bpy.data.objects.new('Camera', camera)
        bpy.context.collection.objects.link(camera_obj)
        bpy.context.view_layer.objects.active = camera_obj
        bpy.context.scene.camera = camera_obj
        camera_obj.location = (0, 0, 0)
        camera_obj.rotation_euler = (math.radians(180), 0, 0)

        # Set up Blender camera parameters to match your self.INTRINSICS
        camera.sensor_fit = 'HORIZONTAL'

        # Calculate sensor width and height based on projector FOV
        projector_fovx = self.INTRINSICS['projector_fovx']
        projector_fovy = self.INTRINSICS['projector_fovy']
        aspect_ratio = math.tan(math.radians(projector_fovx) / 2) / math.tan(math.radians(projector_fovy) / 2)
        camera.sensor_width = 2 * self.INTRINSICS['fx'] * math.tan(math.radians(projector_fovx) / 2)
        camera.sensor_height = camera.sensor_width / aspect_ratio

        # Calculate focal length (in millimeters) from self.INTRINSICS
        focal_length_mm = (self.INTRINSICS['fx'] * camera.sensor_width) / self.width
        camera.lens = focal_length_mm

        # Calculate shift_x and shift_y from self.INTRINSICS
        camera.shift_x = (self.INTRINSICS['cx'] - self.width / 2) / self.width
        camera.shift_y = (self.INTRINSICS['cy'] - self.height / 2) / self.height

        return camera_obj

    def create_point_light(self, location, energy=5000, size=100.0):
        # Create a new point light
        light_data = bpy.data.lights.new("point", 'POINT')
        light_data.energy = energy
        light_data.shadow_soft_size = size
        light_object = bpy.data.objects.new("point", light_data)
        bpy.context.collection.objects.link(light_object)

        # Set the light location
        light_object.location = location

        return light_object

    def calculate_matrix(self, zvec, translation):
        # Normalize the Z vector
        z = np.array(zvec) / np.linalg.norm(zvec)

        # Choose an arbitrary vector not parallel to the Z vector
        if np.allclose(z, np.array([0, 1, 0])):
            arbitrary_vector = np.array([1, 0, 0])
        else:
            arbitrary_vector = np.array([0, 1, 0])

        # Calculate the X vector
        x = np.cross(arbitrary_vector, z)
        x = x / np.linalg.norm(x)

        # Calculate the Y vector
        y = np.cross(z, x)

        # Construct the rotation matrix
        rotation_matrix = np.array([x, y, z]).T

        # Construct the full transformation matrix
        transformation_matrix = np.identity(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation

        return transformation_matrix

    def import_stl(self, filepath):
        # Store the current object names before importing
        current_object_names = set([obj.name for obj in bpy.data.objects])

        # Import the STL file
        bpy.ops.import_mesh.stl(filepath=str(filepath))

        # Find the new object (imported object) by comparing object names
        new_object_names = set([obj.name for obj in bpy.data.objects]) - current_object_names
        if new_object_names:
            obj = bpy.data.objects[new_object_names.pop()]
        else:
            print("No new objects were imported.")
            obj = None

        return obj

    def apply_material(self, obj):
        # Create a new material
        material = bpy.data.materials.new(name='Normal Material')
        obj.active_material = material

    def apply_transformation(self, tm):
        if type(tm) == np.ndarray:
            tm_np = tm
        else:
            tm_np = tm.detach().cpu().numpy()  # Convert the tensor to a NumPy array
        matrix = mathutils.Matrix(tm_np)
        self.obj.matrix_world = matrix

    def render_object(self):
        with stdout_redirected():
            # set path to save the rendered images
            bpy.context.scene.render.filepath = self.output_path

            bpy.ops.render.render(write_still=True, use_viewport=True)

        # load the rendered image and convert it to a numpy array
        img = cv2.imread(self.output_path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img, dtype=np.float32)
        # threshold the image
        img[img < 70] = 0
        img[img >= 255] = 1

        return img

    def render_STL(self, path, centroid, zvec):
        # Calculate the initial transformation matrix
        init_tm = self.calculate_matrix(zvec, centroid)

        # Import the STL file
        self.obj = self.import_stl(path)

        # Apply the material
        self.apply_material(self.obj)

        # Apply the transformation
        self.apply_transformation(init_tm)

        # Render the object and save the image
        self.render_object()

        return init_tm

    def render_iter(self, tm):
        self.iter += 1
        # Apply the transformation
        self.apply_transformation(tm)

        img = self.render_object()

        return img

    def delete(self, scene):
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


import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    TexturesVertex
)


class PyTorch3DRenderer:
    def __init__(self, intrinsics, width, height):
        self.intrinsics = intrinsics
        self.width = width
        self.height = height
        self.mesh = None

        # Compute camera properties
        fov_y = 2 * np.arctan(self.height / (2 * self.intrinsics['fy'])) * (180 / np.pi)
        fov_x = intrinsics['projector_fovx']

        # Create a camera
        self.camera = FoVPerspectiveCameras(
            fov=(fov_y, fov_x),
            aspect_ratio=float(width) / float(height),
            device=torch.device("cuda:0")
        )

        # Create a point light
        self.lights = PointLights(location=[[0.0, 0.0, 0.0]], device=torch.device("cuda:0"))

        # Create a renderer
        raster_settings = RasterizationSettings(image_size=self.width)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.camera, raster_settings=raster_settings),
            shader=SoftSilhouetteShader()
        )

    def load_mesh(self, stl_file):
        return load_objs_as_meshes([stl_file], device=torch.device("cuda:0"))

    def render_mesh(self, tm):
        # Convert transformation matrix to rotation and translation matrices
        R, T = tm[:3, :3], tm[:3, 3]

        # Render the mesh
        self.camera.R = torch.tensor(R, dtype=torch.float32, device=torch.device("cuda:0")).unsqueeze(0)
        self.camera.T = torch.tensor(T, dtype=torch.float32, device=torch.device("cuda:0")).unsqueeze(0)
        images = self.renderer(self.mesh, cameras=self.camera, lights=self.lights)
        binary_mask = images[..., 3].detach().cpu().numpy()

        return binary_mask

    def render(self, stl_file, tm):
        self.mesh = self.load_mesh(stl_file)
        binary_mask = self.render_mesh(tm)
        return binary_mask
