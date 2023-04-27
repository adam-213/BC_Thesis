import pathlib
import random

import scipy.spatial.transform as sst
import torch
from f_PoseRefinment_bpy import *
import f_PoseRefinment_Utils as utils
import matplotlib.pyplot as plt
import cv2
import numpy as np
from stl import mesh
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


####### Simulated Annealing #######
def mask_iou(mask1, mask2):
    """Calculates the intersection over union of two masks"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)


def random_rotvec(perturbation_scale):
    return np.random.uniform(-perturbation_scale, perturbation_scale, (3,))


def perturb_transformation_matrix(current_tm, momentum_tm=None, perturbation_scale=10.0, momentum=0.1):
    # generate the components for a random small delta matrix
    delta_rotvec = random_rotvec(perturbation_scale * 0.5)
    delta_translation = np.random.uniform(-perturbation_scale, perturbation_scale, (3,))

    # convert the delta rotation vector to a rotation matrix
    delta_rotmat = sst.Rotation.from_rotvec(delta_rotvec).as_matrix()

    # join them to a delta transformation matrix
    delta_tm = np.eye(4)
    delta_tm[:3, :3] = delta_rotmat
    delta_tm[:3, 3] = delta_translation

    # apply momentum to the rotation
    if momentum_tm is not None:
        delta_tm[:3, :3] *= (1 + momentum)

    # # apply momentum to the perturbation
    # if momentum_tm is not None:
    #     delta_tm[:3, 3] += momentum * momentum_tm[:3, 3]

    # apply the delta transformation matrix to the current transformation matrix
    new_tm = np.dot(current_tm, delta_tm)

    return new_tm


def acceptance_probability(current_iou, new_iou, temperature, momentum, prev_iou):
    delta_iou = new_iou - current_iou
    momentum_term = momentum * (current_iou - prev_iou)
    if delta_iou > 0:
        return 1.0, delta_iou
    else:
        return 0.0, delta_iou
        return np.exp((delta_iou) / temperature), delta_iou


def refine_SA(mask, init_tm, iters=50, init_temperature=5, cooling_rate=0.9, renderer=None):
    current_tm = init_tm
    best_tm = init_tm

    render_mask = utils.load_image()
    best_iou = mask_iou(mask, render_mask)
    current_iou = best_iou

    temperature = (1 - best_iou) * 5
    scale = (1 - best_iou)
    last_iou = current_iou
    momentum_tm = None

    for iteration in range(iters):
        # scale = 10 * (iters - iteration) / iters

        new_tm = perturb_transformation_matrix(current_tm, momentum_tm=momentum_tm, perturbation_scale=scale)
        scale = scale * 0.95
        # Apply the new transformation matrix and render the mask
        renderer.render_iter(new_tm)
        render_mask = utils.load_image()

        # apply thresholding
        cv2.threshold(render_mask, 100, 255, cv2.THRESH_BINARY, render_mask)

        plt.imshow(render_mask, alpha=0.5)
        plt.imshow(mask, alpha=0.5)
        plt.savefig(f'E:/iter_{iteration}.png')
        plt.close()

        new_iou = mask_iou(mask, render_mask)

        prob, delta_iou = acceptance_probability(current_iou, new_iou, temperature, momentum=0.1, prev_iou=last_iou)
        print(new_iou)
        if best_iou > 0.95:
            print(f'Best IOU: {best_iou}')
        choice = random.choices([0, 1], weights=[1 - prob, prob])[0]

        last_iou = current_iou

        if choice:
            momentum_tm = np.eye(4)
            momentum_tm[:3, 3] = current_tm[:3, 3] - new_tm[:3, 3]
            current_tm = new_tm
            current_iou = new_iou

        if current_iou > best_iou:
            best_tm = current_tm
            best_iou = current_iou

        temperature *= cooling_rate

    print(f'Best IOU: {best_iou}')
    return best_tm, best_iou


def main():
    # setup paths
    base_path = pathlib.Path("E:\Docs_Uniba\BP")
    stl_path = base_path.joinpath('STL')

    renderer = STL_renderer(utils.INTRINSICS, 1032, 772)

    # get the data from the networks - using traning data for now
    z_vec, world_centroid, mask, label, image, ground_truth_matrix, INTRINSICS = utils.get_prediction()
    # render the baseline stl
    init_tm = renderer.render_STL(stl_path.joinpath(f'{label}.stl'), world_centroid, z_vec)
    if label == "part_truster":
        init_tm[2, 3] += 15  # add an offset to the z axis because of the how i acquire the z from the data

    poses = [refine_SA(mask, init_tm, iters=50, init_temperature=5, cooling_rate=0.9, renderer=renderer) for _ in
             range(4)]

    # get the best pose by the iou
    pose = poses[np.argmax([pose[1] for pose in poses])][0]

    return pose, init_tm, ground_truth_matrix, stl_path.joinpath(f'{label}.stl')


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def plot_mesh(fig, stl_mesh, matrix, color, name):
    matrix = np.asarray(matrix)  # Convert the input matrix to a numpy array
    transformed_vertices = np.matmul(matrix[:3, :3], stl_mesh.vectors.reshape(-1, 3).T).T + matrix[:3, 3].reshape(1,
                                                                                                                  -1)
    transformed_faces = transformed_vertices.reshape(-1, 3, 3)
    x, y, z = transformed_faces.reshape(-1, 3).T
    num_triangles = len(x) // 3
    I, J, K = np.arange(0, num_triangles * 3, 3), np.arange(1, num_triangles * 3, 3), np.arange(2,
                                                                                                num_triangles * 3,
                                                                                                3)
    fig.add_trace(
        go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, color=color, opacity=0.5, name=name, showlegend=True)
    )
    return fig


if __name__ == '__main__':
    # set_seeds(5)
    pose, init_tm, gttm, stlpath = main()

    # Load your STL file
    stl_file = str(stlpath)
    print(stl_file)
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # Define your transformation matrices
    matrix1 = np.matrix(pose)
    matrix2 = np.matrix(init_tm)
    matrix3 = np.matrix(gttm)

    fig = go.Figure()

    # Plot the STL file three times with different transformations
    plot_mesh(fig, stl_mesh, matrix1, color='red', name='Pose')
    plot_mesh(fig, stl_mesh, matrix2, color='green', name='Init TM')
    plot_mesh(fig, stl_mesh, matrix3, color='blue', name='GT TM')

    # turn on the legend
    fig.update_layout(showlegend=True)

    fig.show()
