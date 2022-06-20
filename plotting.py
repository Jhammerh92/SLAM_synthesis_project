import numpy as np
# import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation
# import copy
import os
import sys

from load_paths import *

# to plot KITTI pose data
def get_pose_ground_truth_KITTI(file_path):
    pose_data_file = open(file_path,'r')
    pose_data = pose_data_file.readlines()
    position = []
    orientation = []
    # kitti_rot_mat = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02]).reshape((3,3))
    kitti_rot_mat = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
    # print(kitti_rot_mat)
    # print(test)

    for pose in pose_data:
        pose_matrix = np.array([float(s) for s in  str.split(pose)])
        pose_matrix = pose_matrix.reshape((3,4))
        pos = pose_matrix[:,3]
        orien = pose_matrix[:3,:3] 
        position.append(kitti_rot_mat.T @ pos)#([pos[2], -pos[0], -pos[1]])
        orientation.append(orien)
    pose_data_file.close()
    return np.asarray(position), np.asarray(orientation)


def get_shifted_ground_truth(position, orientation,n, first_index=0,):
    # N = len(position)
    # first_index = int(round(N - 2*n ))
    orientation = orientation[first_index::2,:]
    orientation = orientation[:n,:]

    position = position[first_index::2,:3] - position[first_index,:]
    # position = position[::2,:]
    position = position[:n,:]
    return position, orientation

    
def get_pose_ground_truth_DTU(file_path, heading_correction=0.0):
    # pose_data_file = open(file_path,'r')
    pose_data = np.genfromtxt(file_path, skip_header=25, skip_footer=4, usecols=range(2,8), dtype=np.float64)

    # heading_correction = -150.75 # seq DTU 01= -150.75
    init_heading = pose_data[0,3]
    heading_correction_transformation = Rotation.from_euler('zyx', np.array([init_heading, 0.0, 0.0]), degrees=True).as_matrix()
    correction_transformation = Rotation.from_euler('zyx', np.array([heading_correction, 0.0, 0.0]), degrees=True).as_matrix()
    position = pose_data[:,:3]
    position = (correction_transformation @ heading_correction_transformation.T @ (position).T).T
    orientation = Rotation.from_euler('yzx', pose_data[:,3:] - np.array([init_heading, 0.0, 0.0]), degrees=True).as_matrix()

    return np.asarray(position), np.asarray(orientation)


def plot_odometry_2D(pose, axes=None, arg='', label='', **kwargs):
    # 2D plot of egomotion path
    c = pose
    if arg == "origo":
        x = [pt[0]-c[0][0] for pt in c]
        y = [pt[1]-c[0][1] for pt in c]
        z = [pt[2]-c[0][2] for pt in c]
    elif arg == "end":
        x = [pt[0]-c[-1][0] for pt in c]
        y = [pt[1]-c[-1][1] for pt in c]
        z = [pt[2]-c[-1][2] for pt in c]
    elif arg.lower() == "kitti":
        x = [pt[2] for pt in c]
        y = [-pt[0] for pt in c]
        z = [-pt[1] for pt in c]
    else:
        x = [pt[0] for pt in c]
        y = [pt[1] for pt in c]
        z = [pt[2] for pt in c]

    fig = None
    if axes is None:
        fig, axes = plt.subplots(1,2, constrained_layout=True)

    axes[0].scatter(x[0], y[0], s=50, color='g', marker='x') # start
    axes[0].scatter(x[-1], y[-1], s=50, color='r', marker='x') # end
    axes[0].plot(x, y, label=label, **kwargs)
    # axes[0].set_aspect('equal', adjustable='box')
    # axes[0].set_aspect('equal')
    axes[0].axis('equal')
    axes[0].set_xlabel('X [m]')
    axes[0].set_ylabel('Y [m]')
    axes[0].legend()

    axes[1].scatter(x[0], z[0], s=50, color='g', marker='x') # start
    axes[1].scatter(x[-1], z[-1], s=50, color='r', marker='x') # end
    axes[1].plot(x, z, **kwargs)
    axes[1].axis('equal')
    # axes[1].set_aspect('equal', adjustable='box')
    # axes[1].set_aspect('equal')
    axes[1].set_xlabel('X [m]')
    axes[1].set_ylabel('Z [m]')


    axes[0].set_title('XY-plane')
    axes[1].set_title('XZ-plane, elevation')

    if (fig is None):
        return axes
    else:
        return fig, axes


def plot_odometry_colorbar_2D(odometry, c_values, fig, ax=None, arg='', cmap='summer_r'):
    # 3d plot of egomotion path - OBS. Z and Y axis are switched, but labelled correctly in plot
    c = odometry[:len(c_values), :]
    if arg.lower() == "origo":
        x = [pt[0]-c[0][0] for pt in c]
        y = [-(pt[1]-c[0][1]) for pt in c]
        z = [pt[2]-c[0][2] for pt in c]
    elif arg.lower() == 'kitti':
        x = [pt[0] for pt in c]
        y = [-pt[1] for pt in c]
        z = [pt[2] for pt in c]
    else:
        x = [pt[0] for pt in c]
        y = [pt[1] for pt in c]
        z = [pt[2] for pt in c]
    c_values = np.asarray(c_values)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')

    if ax is None:
        fig, ax = plt.subplots(1)

    # ax.plot3D(x, z, y, label='positon')
    # lnWidth = [40 for i in range(len(speed))]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(c_values.min(), c_values.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, antialiaseds=True)
    # Set the values used for colormapping
    lc.set_array(c_values)
    lc.set_linewidth(7)
    line = ax.add_collection(lc)
    # fig.colorbar(line, ax=ax)

    ax.scatter(x[0], y[0], s=50, color='g')
    ax.scatter(x[-1], y[-1], s=50, color='r')


    # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    fig.colorbar(line, ax=ax)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    # ax.set_zlabel('Y [m] - Height')    
    ax.axis('equal')
    # set_axes_equal(ax)

    # plt.show()