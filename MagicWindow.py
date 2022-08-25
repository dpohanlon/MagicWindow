import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rcParams

from tqdm import tqdm

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

plt.rcParams["figure.figsize"] = (16,16)

import argparse

import numpy as np

import seaborn as sns

import PIL

import h5py

import cv2

import open3d as o3d

from PoissonSolver import solve

def create_relief(image_path, size = (250, 250)):

    '''
    Computes the magic window surface profile by solving the Poisson equation.
    '''

    im = PIL.Image.open(image_path)
    if not (size[0] is None):
        im = im.resize(size)

    image = np.array(PIL.ImageOps.grayscale(im))

    n = 1.5
    z = 3.

    poiss = solve(image, n, z)

    f = h5py.File('poisson.h5', 'w')
    f.create_dataset('poisson', data = poiss)
    f.close()

    plt.contourf(np.rot90(poiss.T), 50)
    plt.contour(np.rot90(poiss.T), 50, colors = 'k')
    plt.savefig('poisson.png', dpi = 300)
    plt.clf()

    # Calculate the inverse (Laplacian) using OpenCV to see if it's worked

    lap = cv2.Laplacian(poiss, cv2.CV_64F)
    plt.imshow(lap[1:-1, 1:-1], cmap = 'Greys')
    plt.savefig('laplacian.png', dpi = 300)
    plt.clf()

    return poiss

def create_mesh(poisson):

    '''
    Create a 3D mesh corresponding to the relief surface using Open3D.
    '''

    nx, ny = poisson.shape[0] - 2, poisson.shape[1] - 2

    # Create a point cloud corresponding to the calculate points of the surface

    x, y = np.mgrid[0:1:1./nx, 0:1:1./ny]
    z = poisson[1:-1, 1:-1]

    v = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)

    # Fill the Open3D containers and calculate normals

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v)

    normals = np.array([[0.0, 0.0, -1.0]] * len(v))

    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.estimate_normals()

    # Estimate the 3D mesh triangles from the point cloud

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, linear_fit=True, n_threads = 1)[0]

    o3d.io.write_triangle_mesh('poisson_mesh.obj', poisson_mesh)

    # Embed the surface in a box to generate the first surface

    b = poisson_mesh.create_box(width = 0.95, height = 0.95, depth = 1)

    # Remove one side of the box
    b.triangles = b.triangles[:-2]

    b.translate([0, 0, -0.05])
    b += poisson_mesh

    o3d.io.write_triangle_mesh('poisson_mesh_with_box.obj', b)

if __name__ == '__main__':

    # I'd like an argument, please
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", type=str, dest="image_path", default="", help="Input image path.")
    argParser.add_argument("-sx", type=int, dest="sx", default=None, help="Pixels in output x dimension.")
    argParser.add_argument("-sy", type=int, dest="sy", default=None, help="Pixels in output y dimension.")

    args = argParser.parse_args()

    relief = create_relief(args.image_path, size = (args.sx, args.sy))

    mesh = create_mesh(relief)
