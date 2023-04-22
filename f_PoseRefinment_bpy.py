import bpy
import numpy as np
import mathutils, math
import random, os


class STL_renderer:
    def __init__(self, INTRINSICS, width, height):
        self.INTRINSICS = INTRINSICS
        self.width = width
        self.height = height
        self.obj = None
        self.output_path = "E:\\temp_data.png"

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
        bpy.context.scene.render.image_settings.file_format = 'PNG'

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
        self.obj.matrix_world = mathutils.Matrix(tm)

    def render_object(self):
        # set the output path
        bpy.context.scene.render.filepath = self.output_path
        bpy.ops.render.render(
            use_viewport=True,
            write_still=True,
            scene=bpy.context.scene.name,
        )

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
        # Apply the transformation
        self.apply_transformation(tm)

        self.render_object()

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
