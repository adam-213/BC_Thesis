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
from stl import mesh
from sklearn.neighbors import NearestNeighbors


class ICP_PoseRefinement:

    def __init__(self, init_tm, stl_path, ptc, mask):
        self.losses = []
        self.object = None
        self.tm = init_tm
        self.stl_path = stl_path
        self.ptc = ptc
        self.mask = mask

    def prepare(self):
        self.ptc = self.ptc * self.mask
        # fit a crop to the point cloud for cropping the stl
        coords,center = utils.fit_crop(self.ptc)
        # crop the point cloud
        self.ptc = utils.crop_ptc(self.ptc, coords)
        # load the stl and convert to ply
        self.object = self.stl2ply()

    def stl2ply(self):
        # load stl
        stl_mesh = mesh.Mesh.from_file(self.stl_path)
        # convert to ply
        plydata = mesh.Mesh(stl_mesh.data.copy())

        return plydata

    def run(self):
        self.prepare()
        self.icp()

    def icp(self):
        # Set up initial parameters
        max_iterations = 50
        threshold = 1e-5
        prev_error = 0

        # Iterate until convergence or max_iterations reached
        for _ in range(max_iterations):
            # Find nearest neighbors between point cloud and object mesh
            neigh = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.object.points)
            distances, indices = neigh.kneighbors(self.ptc)

            # Compute the mean error
            error = np.mean(distances)
            self.losses.append(error)

            # Check for convergence
            if np.abs(prev_error - error) < threshold:
                break

            prev_error = error

            # Compute centroids
            centroid_ptc = np.mean(self.ptc, axis=0)
            centroid_obj = np.mean(self.object.points[indices], axis=0)

            # Move point clouds to origin
            ptc_centered = self.ptc - centroid_ptc
            obj_centered = self.object.points[indices].reshape(-1, 3) - centroid_obj

            # Compute rotation matrix
            H = np.dot(ptc_centered.T, obj_centered)
            U, _, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            # Compute translation vector
            t = centroid_obj - np.dot(centroid_ptc, R)

            # Update the transformation matrix
            self.tm[:3, :3] = np.dot(R, self.tm[:3, :3])
            self.tm[:3, 3] += t

            # Apply transformation
            self.ptc = np.dot(self.ptc, R.T) + t


def plot_loss(self):
    plt.plot(self.losses)
    plt.show()


def main():
    # setup paths
    base_path = pathlib.Path("E:\Docs_Uniba\BP")
    stl_path = base_path.joinpath('STL')

    # get the data from the networks - using traning data for now
    z_vec, world_centroid, mask, label, image, ground_truth_matrix, ptc, INTRINSICS = utils.get_prediction()

    # detach and convert to numpy
#    mask = mask.detach().cpu().numpy()
    ptc = ptc.detach().cpu().numpy()
    #image = image.detach().cpu().numpy()


    init_tm = utils.calculate_matrix(z_vec, world_centroid)

    # run the icp
    icp = ICP_PoseRefinement(init_tm, stl_path.joinpath(f'{label}.stl'), ptc, mask)

    icp.run()
    pose = icp.tm

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
