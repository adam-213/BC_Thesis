import pathlib
import random
import torchviz
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
from torch.autograd import Variable
import torch.optim as optim
import logging
import torch.nn as nn
from scipy.optimize import minimize
import queue
import threading
import time


class Refiner:
    def __init__(self, mask, init_tm, return_queue, input_queue, image):
        self.gt_mask = mask
        self.losses, self.tms = [], []

        self.input_queue = input_queue
        self.return_queue = return_queue

        self.depthmap = image[:, :, 3]

        self.rot_scale, self.move_scale, self.combined = None, None, None
        self.prepare_combined(init_tm)

    def prepare_combined(self, init_tm):
        self.gt_mask[self.gt_mask < 0.5] = 0
        self.gt_mask[self.gt_mask >= 0.5] = 1
        # rotvec is in axis angle
        rot, move, scale = utils.decompose_matrix(init_tm)

        # both vectors need to be normalized for the optimizer, however scales need to be saved to rescale for the renderer
        self.rot_scale = np.linalg.norm(rot)
        self.move_scale = np.linalg.norm(move)
        # normalize the vectors
        rotvec = rot / self.rot_scale
        movevec = move / self.move_scale
        # combine the vectors for the optimizer
        self.combined = torch.from_numpy(
            np.array(
                [rotvec[0], rotvec[1], rotvec[2],
                 movevec[0], movevec[1], movevec[2]])).float()

    def calculate_iou_loss(self, mask):
        intersection = np.logical_and(self.gt_mask, mask)
        union = np.logical_or(self.gt_mask, mask)

        # Compute the area of intersection and union
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)

        # Compute the Mask IoU
        iou = intersection_area / float(union_area)

        return 1 - iou

    def calculate_depth_loss(self, mask):
        # mask the depthmap with the mask
        masked_depthmap = self.depthmap * mask
        # exlude the 0 values
        masked_depthmap = masked_depthmap[masked_depthmap != 0]

        # calculate the mean of the depthmap
        mean_depth = np.mean(masked_depthmap)
        # calculate mse between mean and the predicted depth
        depth = self.combined[5]
        return abs(depth - mean_depth)

    def calculate_loss(self, weights):  # render the new mask and calculate the loss
        message = {'type': "render", "args": [weights, self.rot_scale, self.move_scale]}
        self.input_queue.put(message)
        #print("Waiting for render...")
        while self.return_queue.empty():
            # print("Waiting for render...")
            time.sleep(0.01)
        new_mask = self.return_queue.get()
        #print("Got render")

        # turn the mask into bw
        new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)
        new_mask[new_mask < 20] = 0
        new_mask[new_mask >= 20] = 1

        loss_iou = self.calculate_iou_loss(new_mask)
        loss_depth = self.calculate_depth_loss(new_mask)

        print("Loss iou: ", loss_iou)
        print("Loss depth: ", loss_depth)

        # calculate the total loss
        loss =  loss_iou #* 15 + loss_depth  # this needs to be weighted because depth loss is unbounded and iou is bounded between 0 and 1

        #print(loss.item())
        return loss

    def optimize(self, iters=100):
        print("Optimizing...")
        time.sleep(1)

        def wrapper(weights):
            print(weights)
            return self.calculate_loss(weights)

        # Convert the initial combined tensor to a NumPy array
        x0 = self.combined.numpy()

        # Perform the optimization using the Nelder-Mead algorithm
        res = minimize(wrapper, x0, method='Nelder-Mead',
                       options={'maxiter': iters, 'disp': True, 'xatol': 1e-3, 'fatol': 1e-4,
                                'adaptive': True})

        # Update the combined tensor with the optimized weights
        self.combined = torch.tensor(res.x, dtype=torch.float32)

        tm = utils.compose_matrix(self.combined, self.rot_scale, self.move_scale)
        self.result_tm = tm
        self.input_queue.put({'type': "exit", "args": []})
        return tm


def plot_loss(self):
    plt.plot(self.losses)
    plt.show()


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


def main():
    # setup paths
    base_path = pathlib.Path("E:\Docs_Uniba\BP")
    stl_path = base_path.joinpath('STL')

    # get the data from the networks - using traning data for now
    z_vec, world_centroid, mask, label, image, ground_truth_matrix, INTRINSICS = utils.get_prediction()

    init_tm = calculate_matrix(z_vec, world_centroid)

    # init_tm[2, 3] += 5  # add an offset to the z axis because of the how i acquire the z from the data

    input_queue, return_queue = queue.Queue(), queue.Queue()

    refiner = Refiner(mask, init_tm, return_queue, input_queue, image)
    refiner_thread = threading.Thread(target=refiner.optimize)
    refiner_thread.start()

    viewer = STLViewer(utils.INTRINSICS, 1032, 772, stl_path.joinpath(f'{label}.stl'), input_queue, return_queue)

    viewer.apply_transformation(init_tm)
    viewer.mainloop()

    refiner_thread.join()
    pose = refiner.result_tm

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
    # set_seeds(1)
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
    fig = plot_mesh(fig, stl_mesh, matrix1, color='red', name='Pose')
    plot_mesh(fig, stl_mesh, matrix2, color='green', name='Init TM')
    plot_mesh(fig, stl_mesh, matrix3, color='blue', name='GT TM')

    fig.show()
