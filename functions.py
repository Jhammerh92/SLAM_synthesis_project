import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys
import pyquaternion
from scipy.spatial.transform import Rotation

from load_paths import *

# draws two pcds with the source transformed to check alignment of a ICP registration 
def draw_registration_result(target, source, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.7, 0])
    target_temp.paint_uniform_color([0, 0.65, 0.9])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp,target_temp])



def load_view_point(filename, vis=None):
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    ctr.convert_from_pinhole_camera_parameters(param)
    # vis.run()
    # vis.destroy_window()


def set_view_point(vis, T=None, R=None, t=None):
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    extrinsic_matrix = np.eye(4)
    if T is None:
        if not (R is None):
            extrinsic_matrix[:3,:3] = R
        else:
            extrinsic_matrix[:3,:3] = param.extrinsic[:3,:3]
        if not (t is None):
            extrinsic_matrix[:3,3] = t + param.extrinsic[:3,3]
        else:
            extrinsic_matrix[:3,3] = param.extrinsic[:3,3]
    else:
        extrinsic_matrix = T
    new_param = o3d.camera.PinholeCameraParameters()
    new_param.extrinsic = extrinsic_matrix
    new_param.intrinsic = param.intrinsic
    ctr.convert_from_pinhole_camera_parameters(new_param)



def transform_view_point(vis,odometry):
    if not hasattr(transform_view_point, "cam_pose"):
        transform_view_point.param = vis.get_view_control().convert_to_pinhole_camera_parameters() # it doesn't exist yet, so initialize it
        transform_view_point.cam_pose = transform_view_point.param.extrinsic
    cam_odometry = copy.deepcopy(odometry)
    cam_odometry = np.linalg.inv(cam_odometry)
    extrinsic_matrix = transform_view_point.cam_pose @ cam_odometry
    
    new_param = o3d.camera.PinholeCameraParameters()
    new_param.extrinsic = extrinsic_matrix
    new_param.intrinsic = transform_view_point.param.intrinsic
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(new_param)
    # vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()



def add_geometry(vis, geometry):
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    vis.add_geometry(geometry)
    ctr.convert_from_pinhole_camera_parameters(param)


 


# Does point2plane registration of two point clouds. Get the transformation as reg.transformation.
def estimate_p2pl(source, target, init_trans=None, threshold=0.5, use_coarse_estimate=False, return_info=False, max_iteration=30):
    if init_trans is None:
        init_trans=np.array([[1.0, 0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])

    # makes a rough initial estimate, this will help of the actual initial transform is far from the converged result
    if use_coarse_estimate:
        coarse_threshold = 10 * threshold
        coarse_reg = estimate_p2pl(source, target, init_trans, coarse_threshold, max_iteration=100)
        init_trans = coarse_reg.transformation

    reg_p2p = o3d.pipelines.registration.registration_icp(target, source, threshold, init_trans,  # source and target are reversed to get the correct transform
                                                            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    # might need to return this information matrix to make a proper pose graph
    information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, threshold, reg_p2p.transformation)
    if return_info:
        return reg_p2p, information_matrix

    return reg_p2p




# spins the init transform by x deg to find a good init transform at loop closure situations, where the angle is typycally way off
def estimate_p2pl_spin_init(source, target, threshold=0.5, return_info=False, max_iteration=30, spin_axis="z", deg_interval=10):
    # if init_trans is None:
    init_trans=np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

    fitnesses = []
    rot_trans_init = []
    coarse_threshold = 2.5 #5 * threshold
    for i in range(360 // deg_interval):
        rot = Rotation.from_rotvec([0, 0, np.deg2rad(i*deg_interval)])
        # print(rot.as_matrix())
        init_trans[:3,:3] = rot.as_matrix()
        coarse_reg = estimate_p2pl(source, target, init_trans, coarse_threshold, max_iteration=5)
        est_init_trans = coarse_reg.transformation
        # print(coarse_reg.fitness)
        fitnesses.append(coarse_reg.fitness)
        rot_trans_init.append(est_init_trans)

    best_fit_idx = np.argmax(fitnesses)
    best_init_trans = rot_trans_init[best_fit_idx]
    # draw_registration_result(source, target, best_init_trans)

    
    reg_p2p = o3d.pipelines.registration.registration_icp(target, source, threshold, best_init_trans,  # source and target are reversed to get the correct transform
                                                            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )

    # might need to return this information matrix to make a proper pose graph
    information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, threshold, reg_p2p.transformation)
    if return_info:
        return reg_p2p, information_matrix

    return reg_p2p





def save_processed_data(**kwargs):
    temp_cwd = os.getcwd()
    os.chdir(path_to_cwd) # need to save in a location that makes sense

    for key,value in kwargs.items():
        if isinstance(value, np.ndarray):
            np.save(key, value)
        else:
            try:
                o3d.io.write_point_cloud(key + ".pcd", value)
            except:
                print("Can't save {} because it is neither a np.array or pcd".format(key))
    # np.save('O',kwargs['O'])

    # create a .txt with description

    os.chdir(temp_cwd)




def load_pcd_from_index(index, path="", is_downsampled=False, voxel_size=0.5):
    pcd_name = path + "{:06d}.pcd".format(index)
    pcd = o3d.io.read_point_cloud(pcd_name)
    if is_downsampled:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd



def calculate_internal_distances(points, xy_only=False):
    sq_distance_matrix = np.empty((len(points),len(points)))
    points_d = copy.deepcopy(points)
    if xy_only: # only consider distnace in x and y, ignoring elevation difference.
        points_d = points_d[:,:2]
    for i, point in enumerate(points_d):
        sq_distances = np.sum((point - points_d)**2, axis=1) # keeping them sqaured to avoid root
        # sq_distances = np.linalg.norm((point - points),2, axis=1) # the actual distance
        sq_distance_matrix[i,:] = sq_distances
    return sq_distance_matrix


# a = np.random.normal(size=(3,3))
# d = calculate_internal_distances(a)
# print(d)


def translation_dist_of_transform(transformation, timestep=1.0):
    distance = np.linalg.norm(transformation[:3,3], 2)
    return distance/timestep # in case timestep is given the distance is the instantaneous speed


def angular_rotation_of_transform(transformation, timestep=1.0):
    angle = pyquaternion.Quaternion(matrix=transformation[:3, :3]).degrees
    return angle/timestep # in case timestep is given the anglee is the instantaneous angular speed





# to plot KITTI pose data
def get_pose_ground_truth_KITTI(file_path):
    pose_data_file = open(file_path,'r')
    pose_data = pose_data_file.readlines()
    position = []
    orientation = []
    for pose in pose_data:
        pose_matrix = np.array([float(s) for s in  str.split(pose)])
        pose_matrix = pose_matrix.reshape((3,4))
        pos = pose_matrix[:,3]
        orien = pose_matrix[:3,:3]
        position.append(pos.T)
        orientation.append(orien)
    pose_data_file.close()
    return np.asarray(position), np.asarray(orientation)



def plot_odometry_2D(pose, ax, arg=None):
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
    else:
        x = [pt[0] for pt in c]
        y = [pt[1] for pt in c]
        z = [pt[2] for pt in c]

    ax.scatter(x[0], z[0], s=50, color='g') # start
    ax.scatter(x[-1], z[-1], s=50, color='r') # end

    ax.plot(x, z, label='pose odometry')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m] - Depth')


