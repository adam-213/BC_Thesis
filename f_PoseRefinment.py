import pathlib
import random
import torchviz
import scipy.spatial.transform as sst
import torch
import f_PoseRefinment_Utils as utils
import matplotlib.pyplot as plt
import cv2
import numpy as np
from stl import mesh
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from stl import mesh
from sklearn.neighbors import NearestNeighbors

import open3d as o3d
import numpy as np
import pathlib
import random
import torch
import f_PoseRefinment_Utils as utils


class ICP_PoseRefinement:
    def __init__(self, init_tm, stl_path, ptc):
        self.tm = init_tm
        self.stl_path = str(stl_path)
        self.ptc = o3d.geometry.PointCloud()
        self.ptc.points = o3d.utility.Vector3dVector(ptc)

    def prepare(self):
        #self.ptc = self.ptc.voxel_down_sample(voxel_size=0.02)

        # load the stl and convert to point cloud
        self.object = o3d.io.read_triangle_mesh(self.stl_path)
        self.object = self.object.sample_points_poisson_disk(number_of_points=3000)

    def run(self):
        self.prepare()
        return self.icp()

    def icp(self):
        # Set up initial parameters
        threshold = 150
        max_iter = 1000


        # Convergence criteria
        convergence_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter,
                                                                                 relative_fitness=1e-1,
                                                                                 relative_rmse=1e-1,)

        # Run ICP
        reg_p2l = o3d.pipelines.registration.registration_icp(
            self.ptc, self.object, threshold, self.tm,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            convergence_criteria)

        # check convergence
        print(reg_p2l)

        return reg_p2l.transformation



def plot_loss(self):
    plt.plot(self.losses)
    plt.show()


def main():
    # setup paths
    base_path = pathlib.Path("E:\Docs_Uniba\BP")
    stl_path = base_path.joinpath('stl')

    # get the data from the networks - using traning data for now
    z_vec, centroid, mask, label, image, ground_truth_matrix, ptc, INTRINSICS, world_centroid = utils.get_prediction()

    # detach and convert to numpy
    #    mask = mask.detach().cpu().numpy()

    ptc = ptc.detach().cpu().squeeze(0).numpy()
    # image = image.detach().cpu().numpy()
    world_centroid = list(world_centroid)
    world_centroid[2] += 5

    init_tm = utils.calculate_matrix(z_vec, world_centroid)

    # mask the ptc
    ptc = ptc * mask

    # cut to the mask bbox
    ptc = utils.cut_to_bbox(ptc, mask)


    ptc = ptc.transpose(1, 2, 0).reshape(-1, 3)

    # run the icp
    icp = ICP_PoseRefinement(init_tm, stl_path.joinpath(f'{label}.stl'), ptc)

    pose = icp.run()

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
    # print difference between the two
    print(f'Pose: {pose}')
    print(f'Init TM: {init_tm}')

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
