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
from torch.autograd import Variable
import torch.optim as optim
import logging

logging.getLogger("bpy").disabled = True


class Refiner:
    def __init__(self, mask, init_tm, renderer):
        self.gt_mask = torch.from_numpy(mask).float()
        self.renderer = renderer
        self.losses = []
        self.tms = []

        self.image_path = "E:/temp_data.png"
        # rotvec is in axis angle
        self.rot_vec, self.trans_vec, self.scale_vec = utils.decompose_matrix(init_tm)

        # both vectors need to be normalized for the optimizer, however scales need to be saved to rescale for the renderer
        self.rot_vec_scale = np.linalg.norm(self.rot_vec)
        self.rot_vec = self.rot_vec / self.rot_vec_scale
        self.trans_vec_scale = np.linalg.norm(self.trans_vec)
        self.trans_vec = self.trans_vec / self.trans_vec_scale

        self.combined = torch.from_numpy(np.array(
            [self.rot_vec[0], self.rot_vec[1], self.rot_vec[2], self.trans_vec[0], self.trans_vec[1],
             self.trans_vec[2]]))

        self.combined.requires_grad = True
        self.optimizer = optim.Adam([self.combined], lr=0.0015)

    def calculate_iou_loss(self, mask):  # calculate the intersection over union loss
        intersection = self.gt_mask * mask
        union = self.gt_mask + mask

        intersection = torch.sum(intersection)
        union = torch.sum(union)

        loss = 1 - (intersection / union)
        return loss

    def calculate_loss(self):  # render the new mask and calculate the loss
        tm = utils.compose_matrix(self.combined, self.rot_vec_scale,
                                  self.trans_vec_scale)
        new_mask = self.renderer.render_iter(tm)
        new_mask = torch.from_numpy(new_mask).float()
        loss = self.calculate_iou_loss(new_mask)
        print(loss.item())
        return loss

    def optimize(self, iters=50):
        for i in range(iters):
            self.optimizer.zero_grad()
            loss = self.calculate_loss()
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())

    def plot_loss(self):
        plt.plot(self.losses)
        plt.show()


def main():
    # setup paths
    base_path = pathlib.Path("E:\Docs_Uniba\BP")
    stl_path = base_path.joinpath('STL')

    renderer = STL_renderer(utils.INTRINSICS, 1032, 772)
    # get the data from the networks - using traning data for now
    z_vec, world_centroid, mask, label, image, ground_truth_matrix, INTRINSICS = utils.get_prediction()
    # render the baseline stl
    init_tm = renderer.render_STL(stl_path.joinpath(f'{label}.stl'), world_centroid, z_vec)

    print(init_tm)
    init_tm[2, 3] += 10  # add an offset to the z axis because of the how i acquire the z from the data
    refiner = Refiner(mask, init_tm, renderer)
    pose = refiner.optimize()
    print(renderer.iter)

    print(pose)
    print(ground_truth_matrix)

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
        go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, color=color, opacity=0.5, name=name)
    )
    return fig


if __name__ == '__main__':
    set_seeds(1)
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
