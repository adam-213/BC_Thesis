from f_inference import main as network_main
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import cv2

INTRINSICS = {
    'fx': 1181.077335,
    'fy': 1181.077335,
    'cx': 516.0,
    'cy': 386.0,
    'projector_fovx': 47.50,
    'projector_fovy': 36.00,
}


def world_to_image_coords(world_coords, INTRINSICS):
    fx, fy, cx, cy = INTRINSICS['fx'], INTRINSICS['fy'], INTRINSICS['cx'], INTRINSICS['cy']
    X, Y, Z = world_coords

    # Normalize the real-world coordinates
    x = X / Z
    y = Y / Z

    # Apply the intrinsic parameters to convert to pixel coordinates
    u = fx * x + cx
    v = fy * y + cy

    # Round to integer values
    u, v = round(u), round(v)

    return u, v


def image_to_world_coords(image_coords, INTRINSICS, Z):
    fx, fy, cx, cy = INTRINSICS['fx'], INTRINSICS['fy'], INTRINSICS['cx'], INTRINSICS['cy']
    u, v = image_coords

    # Convert pixel coordinates to normalized image coordinates
    x = (u - cx) / fx
    y = (v - cy) / fy

    # Compute the real-world coord inates
    X = x * Z
    Y = y * Z

    return X, Y, Z


def get_prediction():
    # theoretically, an image should be input here but that is not happening yet
    z_vec, centroid, mask, label, image, ground_truth_matrix, ptc, worldcentroid = network_main()

    centroid = centroid[0].cpu().detach().numpy()
    mask = mask[0].cpu().detach().numpy()
    z_vec = z_vec[0].flatten()

    image = image[:, :, [0, 1, 2, 3]]  # remove the albedo channel, which is only used in mrcnn



    return z_vec, centroid, mask, label, image, ground_truth_matrix, ptc, INTRINSICS, worldcentroid


def decompose_matrix(matrix):
    # Compute scale factors
    scale_vector = np.array(
        [np.linalg.norm(matrix[:3, 0]), np.linalg.norm(matrix[:3, 1]), np.linalg.norm(matrix[:3, 2])])

    # Normalize the rotation matrix columns by the scale factors
    rotation_matrix = np.array(matrix[:3, :3])
    rotation_matrix[:, 0] /= scale_vector[0]
    rotation_matrix[:, 1] /= scale_vector[1]
    rotation_matrix[:, 2] /= scale_vector[2]

    # Compute the axis-angle of the rotation matrix
    r = Rotation.from_matrix(rotation_matrix)
    axis_angle = r.as_rotvec()

    # Extract the translation vector
    translation_vector = matrix[:3, 3]

    return axis_angle, translation_vector, scale_vector


def load_image(path="E:\\temp_data.png"):
    """Loads the grayscale image from the given path"""
    img = Image.open(path)
    img = img.convert('L')
    img = np.array(img)
    return img


def compose_matrix(combined, rot, move):
    if type(combined) == torch.Tensor:
        combined = combined.clone().cpu().detach().numpy()
    axis_angle, translation_vector = combined[:3] * rot, combined[3:] * move

    axis_angle, translation_vector = axis_angle.flatten(), translation_vector.flatten()
    # Create a rotation matrix from the axis-angle
    r = Rotation.from_rotvec(axis_angle)
    rotation_matrix = r.as_matrix()

    # Assemble the  matrix
    new_matrix = np.zeros((4, 4))
    new_matrix[:3, :3] = rotation_matrix
    new_matrix[:3, 3] = translation_vector
    new_matrix[3, 3] = 1

    return new_matrix


def calculate_matrix(zvec, translation):
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


def cut_to_bbox(ptc,mask):
    points = np.stack(np.where(mask), axis=1)
    x, y, w, h = cv2.boundingRect(points)
    return  ptc[:, x:x + h, y:y + w]