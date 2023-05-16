import bpy
import numpy as np
import mathutils, math
import random, os
import cv2
import f_PoseRefinment_Utils as utils
import os
import sys
from contextlib import contextmanager
import torch
import threading

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from stl import mesh
import numpy as np
import stl
import OpenGL.GL as gl
import OpenGL.GLU as glu
import math
from copy import deepcopy
from PIL import Image


@contextmanager
def stdout_redirected(to=os.devnull):
    # blender shut_up-inator
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


class STL_renderer(torch.nn.Module):
    def __init__(self, INTRINSICS, width, height):
        super().__init__()
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
        # bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.resolution_x = self.width
        bpy.context.scene.render.resolution_y = self.height
        # set to cuda on evvee

        # # Set EEVEE settings
        # bpy.context.scene.eevee.use_bloom = False
        # bpy.context.scene.eevee.use_ssr = False
        # bpy.context.scene.eevee.use_soft_shadows = False
        # bpy.context.scene.eevee.taa_render_samples = 2
        # bpy.context.scene.eevee.taa_samples = 1

        # Simplify the scene
        # bpy.context.scene.render.use_simplify = True
        # bpy.context.scene.render.simplify_subdivision_render = 1
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

            bpy.ops.render.opengl(animation=False, sequencer=False, write_still=True)

        # load the rendered image and convert it to a numpy array
        img = cv2.imread(self.output_path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img, dtype=np.float32)
        # threshold the image
        img[img < 70] = 0
        img[img >= 70] = 1

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

    def render_iter(self, *args):
        tm = utils.compose_matrix(*args)
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


class STLViewer:

    def __init__(self, intrinsic, width, height, stl, queue, return_queue):
        self.width = width
        self.height = height
        self.intrinsics = intrinsic
        self.display = (width, height)
        self.vertices = None
        self.normals = None
        self.vertices_base = None
        self.normals_base = None
        self.screen = None
        self.opengl_lock = threading.Lock()

        self.load_stl(stl)

        self.queue = queue
        self.return_queue = return_queue

        self.screen = self.setup_camera()
        self.setup_lighting()
        #self.mainloop()

    def setup_camera(self):
        # Initialize Pygame
        pygame.init()
        size = (self.width, self.height)
        screen = pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL)

        # Initialize OpenGL
        gl.glViewport(0, 0, size[0], size[1])
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(self.intrinsics['projector_fovy'], float(size[0]) / float(size[1]), 0.1, 10000.0)

        # Set up Pygame/OpenGL camera parameters to match intrinsics
        sensor_fit = 'HORIZONTAL'
        projector_fovx = self.intrinsics['projector_fovx']
        projector_fovy = self.intrinsics['projector_fovy']
        aspect_ratio = math.tan(math.radians(projector_fovx) / 2) / math.tan(math.radians(projector_fovy) / 2)
        sensor_width = 2 * self.intrinsics['fx'] * math.tan(math.radians(projector_fovx) / 2)
        sensor_height = sensor_width / aspect_ratio

        focal_length_mm = (self.intrinsics['fx'] * sensor_width) / size[0]
        shift_x = (self.intrinsics['cx'] - size[0] / 2) / size[0]
        shift_y = (self.intrinsics['cy'] - size[1] / 2) / size[1]

        # Position and orient the camera
        camera_pos = (0, 0, 0)
        camera_up = (0, -1, 0)
        camera_look = (0, 0, 1)
        glu.gluLookAt(
            camera_pos[0], camera_pos[1], camera_pos[2],
            camera_look[0], camera_look[1], camera_look[2],
            camera_up[0], camera_up[1], camera_up[2],
        )

        # Add a light behind the camera
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [0, 0, -1, 0])
        gl.glEnable(gl.GL_LIGHT0)

        # Enable depth testing
        #gl.glEnable(gl.GL_DEPTH_TEST)

        return screen

    def setup_lighting(self):
        # Set up a directional light behind the camera
        light_ambient = [0.2, 0.2, 0.2, 1.0]
        light_diffuse = [1.0, 1.0, 1.0, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]
        light_position = [0.0, 0.0, -1.0, 0.0]
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, light_ambient)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, light_diffuse)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, light_specular)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)
        gl.glEnable(gl.GL_LIGHT0)

        # Set up material properties
        mat_ambient = [0.2, 0.2, 0.2, 1.0]
        mat_diffuse = [0.8, 0.8, 0.8, 1.0]
        mat_specular = [1.0, 1.0, 1.0, 1.0]
        mat_shininess = 0.0
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, mat_ambient)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, mat_diffuse)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, mat_specular)
        gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, mat_shininess)

        # Enable lighting and depth testing
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Set the shading model to smooth
        gl.glShadeModel(gl.GL_SMOOTH)

    def load_stl(self, filename):
        mesh = stl.mesh.Mesh.from_file(filename)
        self.vertices = mesh.vectors.reshape(-1, 3)
        self.normals = mesh.normals.reshape(-1, 1)

        self.normals_base = deepcopy(self.normals)
        self.vertices_base = deepcopy(self.vertices)

    def mainloop(self):
        running = True
        while running:
            # Handle events
            if not self.queue.empty():
                message = self.queue.get()
            else:
                message = None

            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         pygame.quit()
            #         quit()

            if message is not None:
                if message["type"] == "exit":
                    pygame.quit()
                    running = False
                    continue
                    #quit()
                if message["type"] == "render":
                    #print("message received")
                    self.apply_transform_iter(*message["args"])
                    #print("message processed")

            # Clear the screen and depth buffer
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                # Draw the STL mesh
            if self.vertices is not None and self.normals is not None:
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                vertex_array = (gl.GLfloat * self.vertices.flatten().size)(*self.vertices.flatten())


                gl.glVertexPointer(3, gl.GL_FLOAT, 0, vertex_array)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(self.vertices))
            # Swap buffers
            pygame.display.flip()
            # limit to 60 fps using pygame
            pygame.time.wait(16)

            if message is not None:
                #print("processing message")
                if message["type"] == "render":
                    image = self.render_object()
                    #print("sending image")
                    self.return_queue.put(image)
                    #print("image sent")

    def apply_transformation(self, matrix):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # Apply the transformation to the vertices
        homogeneous_vertices = np.hstack((self.vertices_base, np.ones((self.vertices.shape[0], 1))))
        transformed_vertices = np.dot(matrix , homogeneous_vertices.T).T
        transformed_vertices = transformed_vertices[:, :3] / transformed_vertices[:, 3][:, np.newaxis]
        self.vertices = transformed_vertices.astype(np.float32)

    def render_object(self):
        # get the image from the buffer
        buffer = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image = Image.frombytes(mode="RGB", size=(self.width, self.height), data=buffer)
        image = np.array(image)
        image = np.flipud(image)

        return image

    def apply_transform_iter(self, *args):
        tm = utils.compose_matrix(*args)
        tm = np.array(tm)
        # Apply the transformation
        print(tm)
        self.apply_transformation(tm)
