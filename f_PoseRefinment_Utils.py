from f_inference import main as network_main
import numpy as np
from PIL import Image

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
    z_vec, centroid, mask, label, image, ground_truth_matrix = network_main()

    centroid = centroid[0].cpu().detach().numpy()
    mask = mask[0].cpu().detach().numpy()
    z_vec = z_vec[0].flatten()

    image = image[:, :, [0, 1, 2, 3]]  # remove the albedo channel, which is only used in mrcnn

    world_centroid = image_to_world_coords(centroid[:2], INTRINSICS, centroid[2])
    world_centroid = [world_centroid[0].item(), world_centroid[1].item(), centroid[2].item()]

    return z_vec, world_centroid, mask, label, image, ground_truth_matrix, INTRINSICS


def decompose_matrix(matrix):
    scale_vector = np.array(
        [np.linalg.norm(matrix[:3, 0]), np.linalg.norm(matrix[:3, 1]), np.linalg.norm(matrix[:3, 2])])
    rotation_matrix = np.array(matrix[:3, :3])
    rotation_matrix[:, 0] /= scale_vector[0]
    rotation_matrix[:, 1] /= scale_vector[1]
    rotation_matrix[:, 2] /= scale_vector[2]
    translation_vector = matrix[:3, 3]
    return rotation_matrix, translation_vector, scale_vector


def load_image(path="E:\\temp_data.png"):
    """Loads the grayscale image from the given path"""
    img = Image.open(path)
    img = img.convert('L')
    img = np.array(img)
    return img


def compose_matrix(rotation_matrix, translation_vector, scale_vector):
    new_matrix = np.zeros((4, 4))
    new_matrix[:3, :3] = rotation_matrix * scale_vector.reshape(3, 1)
    new_matrix[:3, 3] = translation_vector
    new_matrix[3, 3] = 1
    return new_matrix
