import numpy as np
import open3d as o3d
# import matplotlib.pyplot as plt
import copy
import os
import sys
import pyquaternion
from scipy.spatial.transform import Rotation
import re
import time
import pickle

from load_paths import *

# draws two pcds with the source transformed to check alignment of a ICP registration 
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.7, 0])
    target_temp.paint_uniform_color([0, 0.65, 0.9])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp,target_temp],
                                      zoom=0.1,
                                      front=[0.0, 0.4, 0.4],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[0.0, 0.0, 1])

def draw_pcd(source):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().load_from_json(path_to_cwd + 'render_options_vis.json')
    if isinstance(source, list):
        for item in source:
            vis.add_geometry(item)
    else:
        vis.add_geometry(source)
    vis.run()


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



def transform_view_point(vis, odometry):
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
    # vis.poll_events()
    # vis.update_renderer()



def add_geometry(vis, geometry):
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    vis.add_geometry(geometry)
    ctr.convert_from_pinhole_camera_parameters(param)





def extract_ground_plane(pcd_in, threshold, max_iteration=1000):
    pcd = copy.deepcopy(pcd_in)
    pcd = crop_pcd(pcd, z=(-2.5,-1.0))
    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                            ransac_n=3,
                                            num_iterations=max_iteration)

    ground_plane_pcd = pcd.select_by_index(inliers)
    top_pcd = pcd.select_by_index(inliers, invert=True)
    return ground_plane_pcd, top_pcd, inliers, plane_model


def construct_plane_from_model(model=[0,0,0,0], size=10, density=10):
    # create perfect plane in x, y, z=0 in eg. 50x50 meters
    def null(a, rtol=1e-5):
        u, s, v = np.linalg.svd(a)
        rank = (s > rtol*s[0]).sum()
        return rank, v[rank:].T.copy()

    area = (-size, size)
    point_in_plane = np.array([0.0,0.0,-model[3]], ndmin=2)
    normal = np.array([model[0],model[1],model[2]],ndmin=2).T
    rank, Q = null(normal.T)
    X = np.linspace(*area, size*density)
    Y = X
    XX, YY = np.meshgrid(X,Y)
    XX = XX.ravel()
    YY = YY.ravel()
    if rank == 0:
        ZZ = np.zeros_like(XX)
        points = np.c_[XX, YY, ZZ]
    else:
        points = np.c_[XX, YY]
    plane_np = point_in_plane + (Q @ points.T).T
    plane = o3d.geometry.PointCloud()
    plane.points = o3d.utility.Vector3dVector(plane_np)

    plane.estimate_normals(
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 5, max_nn = 10))
    # normals are expected to point toward the Lidar scanner location. 
    plane.orient_normals_to_align_with_direction(np.array([0.0,0.0,1.0]))
    # fin_np = np.c_[ZZ, XX, YY]
    return plane




# Does point2plane registration of two point clouds. Get the transformation as reg.transformation.
def estimate_p2pl(source, target, init_trans=None, threshold=0.5, use_coarse_estimate=False, return_info=False, max_iteration=50):
    if init_trans is None:
        init_trans=np.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])

    # makes a rough initial estimate, this can help if the actual initial transform is far from the converged transform
    if use_coarse_estimate:
        coarse_threshold = 10 * threshold
        coarse_reg = estimate_p2pl(source, target, init_trans, coarse_threshold, max_iteration=15)
        init_trans = coarse_reg.transformation

    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, init_trans,  # source and target are reversed to get the correct transform
                                                            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )

    if return_info:
        information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(target, source, threshold, reg_p2p.transformation)
        evaluation = o3d.pipelines.registration.evaluate_registration(target, source, threshold, reg_p2p.transformation )
        return reg_p2p, information_matrix, evaluation

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





def save_processed_data(SLAM, **kwargs):
    temp_cwd = os.getcwd()

    folders = listdirs(PATH_TO_PROCESSED)
    n = int(folders[-1].split('_')[-1]) +1
    proc_data_folder = f"{SEQUENCE_NAME}_run_{(n):03d}"
    folder_path = f'{PATH_TO_PROCESSED}/{proc_data_folder}'
    os.mkdir(folder_path)
    os.chdir(folder_path) 

    for key,value in kwargs.items():
        if isinstance(value, np.ndarray):
            fmt = '%.6e'
            if key == 'timestamps':
                fmt = '%.32e'
            np.save(key, value)
            np.savetxt(f'{key}.txt', value, fmt=fmt, delimiter=" ")
        else:
            try:
                o3d.io.write_point_cloud(key + ".pcd", value)
            except:
                print("Can't save {} because it is neither a np.array or pcd".format(key))

    #pickle settings

    with open('settings.pickle', 'wb') as handle:
        pickle.dump(SETTINGS, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('pose_graph.pickle', 'wb') as handle:
    #     pickle.dump(SLAM.keyframes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # create list of strings
    with open("settings.txt", 'w') as f: 
        for key, value in SETTINGS_DICT.items(): 
            f.write(f'{key:25}{value}\n')

    print(f"Data saved in {folder_path}")

    # create a .txt with description!!

    os.chdir(temp_cwd)




def load_pcd_from_index(index, path="", is_downsampled=False, voxel_size=0.5, display_load_time=False):
    t_load1 = time.perf_counter()
    if path[-1] != '/':
        path += '/'
    pcd_ls = [file for file in os.listdir(path) if file.endswith('.pcd')]
    pcd_ls.sort(key=lambda x: "{:06d}".format(int(x.replace('.','_').split('_')[0])) )
    # pcd_name = path + "{:06d}.pcd".format(index)
    pcd_name = path + pcd_ls[index]
    pcd = o3d.io.read_point_cloud(pcd_name)
    if is_downsampled:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    t_load2 = time.perf_counter()
    if display_load_time: print(f"load time:\t {t_load2-t_load1:2f}s")
    return pcd



def calculate_internal_distances(points, xy_only=False, sparse=False): # MAKE IT SPARSE!
    sq_distance_matrix = np.empty((len(points),len(points)))
    points_d = copy.deepcopy(points)
    if xy_only: # only consider distnace in x and y, ignoring elevation difference.
        points_d = points_d[:,:2]
    for i, point in enumerate(points_d):
        sq_distances = np.sum((point - points_d)**2, axis=1) # keeping them sqaured to avoid root
        # sq_distances = np.linalg.norm((point - points),2, axis=1) # the actual distance
        sq_distance_matrix[i,:] = sq_distances
    return sq_distance_matrix



def translation_dist_of_transform(transformation, timestep=1.0):
    distance = np.linalg.norm(transformation[:3,3], 2)
    return distance/timestep # in case timestep is given the distance is the instantaneous speed


def angular_rotation_of_transform(transformation, timestep=1.0, is_deg=False, axis=None):
    if axis is None:
        angle = pyquaternion.Quaternion(matrix=transformation[:3, :3])
        angle = angle.degrees if is_deg else angle.radians
    elif axis == 'z':
        angle = -np.arctan2(transformation[1,0], transformation[0,0])
        angle = np.rad2deg(angle) if is_deg else angle

    return angle/timestep # in case timestep is given the anglee is the instantaneous angular speed


def crop_pcd(pcd, x=None,y=None,z=None, r=None):
    points = np.asarray(pcd.points)
   
    mask_x = np.atleast_2d(np.logical_and(points[:,0] > x[0], points[:,0] < x[1])).T if not(x is None) else np.full((len(points),1), True)
    mask_y = np.atleast_2d(np.logical_and(points[:,1] > y[0], points[:,1] < y[1])).T if not(y is None) else np.full((len(points),1), True)
    mask_z = np.atleast_2d(np.logical_and(points[:,2] > z[0], points[:,2] < z[1])).T if not(z is None) else np.full((len(points),1), True)
    mask_r = np.atleast_2d(np.logical_and(np.linalg.norm(points[:,:2], axis=1) > r[0], np.linalg.norm(points[:,:2], axis=1) < r_[1])).T if not(r is None) else np.full((len(points),1), True)

    # m1 = np.logical_and(mask_x , mask_y)
    # m2 = np.logical_and(mask_z , mask_r)
    mask = np.logical_and.reduce([mask_x , mask_y, mask_z , mask_r])
    # mask = mask_x & mask_y & mask_z & mask_r
    pcd = pcd.select_by_index(np.where(mask)[0])
    return pcd



def pcd_create_normals(path):
    if path[-1] != '/':
        path += '/'
    pcd_ls = [file for file in os.listdir(path) if file.endswith('.pcd')]
    pcd_ls.sort(key=lambda x: "{:06d}".format(int(x.split('_')[0])) )
    # pcd_name = path + "{:06d}.pcd".format(index)
    for index in range(len(pcd_ls)):
        pcd_name = path + pcd_ls[index]
        pcd = o3d.io.read_point_cloud(pcd_name)
        pcd.estimate_normals(
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 100))
        # normals are expected to point toward the Lidar scanner location. 
        pcd.orient_normals_towards_camera_location(np.array([0,0,2.0]))
        o3d.io.write_point_cloud(path + pcd_name, pcd)




def create_vertex_map(pcd):
    object_points = np.asarray(pcd.points)

    # fx = 1 # for 360 (actually 180) degree fov
    # fy = 90.80712071513639 # for 26.8 degree fov
    # cx = 0.0
    # cy = 0.0
    # cameraMatrix = np.array([[fx, 0,0],[ 0,fy, 0],[cx,cy,1]])
    # image_points, _ = cv2.projectPoints(object_points, np.eye(3), np.zeros((3,1)), cameraMatrix, 0)

    depth = np.linalg.norm(object_points, axis = 1)
    pitch = np.arcsin(object_points[:,2] / depth)
    yaw = np.arctan2(object_points[:,1], object_points[:,0])

    FOV = (np.max(pitch)- np.min(pitch)) #np.deg2rad(26.8)
    FOV_down = np.min(pitch)# np.deg2rad(-24.2)

    # FOV_down = np.deg2rad(-26.8)
    # FOV_up = np.deg2rad(2.0)
    # FOV = FOV_up + abs(FOV_down)

    h = 64 # 16
    w = 2048 # 1200
    # w = 1024

    u = (w) * (1/2 * (1 - yaw / np.pi))
    v = (h-1) * (1 - (pitch - FOV_down) / FOV)

    # vertex_map = np.full((row_scale, col_scale), np.nan)
    # vertex_map = [[0 for _ in range(w)] for _ in range(h)]
    # for u,v, d in zip(u.astype(np.int64),v.astype(np.int64), depth):
    #     vertex_map[ v][ u ] = d

    vertex_map = np.zeros((h, w)) # KAN DETTE GØRES UDEN ET LOOP?!
    for u,v, d in zip(u.astype(np.int64),v.astype(np.int64), depth):
        vertex_map[v, u] = d

    # @jit(cache=True, nopython=True)
    # def iterate_image_points(U,V,d):
    #     # vertex_map = np.full((row_scale, col_scale), np.nan)
    #     vertex_map = np.zeros((h, w))
    #     for i in range(len(V)):
    #         vertex_map[V[i], U[i]] = d[i]
    #     return vertex_map

    # vertex_map = iterate_image_points(u.astype(np.int64), v.astype(np.int64), depth)
    return vertex_map


def create_location_histogram(pcd, density = 0.1, area=10):
    n = int((2*area/density) // (density/density))
    bins = [np.linspace(-area,area,n), np.linspace(-area,area,n)]
    points = np.asarray(pcd.points)
    location_histogram = np.histogram2d(points[:,0], points[:,1], bins=bins)
    return location_histogram[0]


# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(rotation_matrix) :
    Rt = np.transpose(rotation_matrix)
    shouldBeIdentity = np.dot(rotation_matrix.T, rotation_matrix)
    I = np.identity(3, dtype = rotation_matrix.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotation_matrix2euler_angles(rotation_matrix) :
    assert(is_rotation_matrix(rotation_matrix))
    R = rotation_matrix
    sy = np.sqrt(R[0,0]**2 +  R[1,0]**2)
 
    singular = sy < 1e-6
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


def denoise_pcd(pcd):
    cl, denoised_ind = pcd.remove_statistical_outlier(nb_neighbors=6, std_ratio=2.0)
    denoised_cloud = pcd.select_by_index(denoised_ind)
    noise_cloud = pcd.select_by_index(denoised_ind, invert=True)

    return denoised_cloud



def ground_plane_constrained_ICP(source, target, init_trans=np.eye(4), max_iteration=50, return_info=False, threshold=0.2):
    source_ground_plane, source_top, source_inliers, source_model = extract_ground_plane(source, 0.1, max_iteration=100)
    target_ground_plane, target_top, target_inliers, target_model = extract_ground_plane(target, 0.1, max_iteration=100)


    source_plane = construct_plane_from_model(source_model,size=30, density=1)
    target_plane = construct_plane_from_model(target_model,size=30, density=1)
    # source_top = copy.deepcopy(source)
    # target_top = copy.deepcopy(target)
    # source_top.select_by_index(source_inliers, invert=True)
    # target_top.select_by_index(target_inliers, invert=True)

    # draw_pcd([source_plane, source_ground_plane])
    # draw_pcd(source_top)
    
    # reg_top = o3d.pipelines.registration.registration_icp(target_top, source_top, threshold, init_transformation,  # source and target are reversed to get the correct transform
    #                                                     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #                                                     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    reg_top = estimate_p2pl(source, target, use_coarse_estimate=True, init_trans=init_trans)

    reg_plane = o3d.pipelines.registration.registration_icp(target_plane, source_plane, threshold, np.eye(4),  # source and target are reversed to get the correct transform
                                                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    # draw_registration_result(source_top, target_top, reg_top.transformation)

    rot = copy.deepcopy(reg_top.transformation[:3,:3])
    r = Rotation.from_matrix(rot)
    euler_ang_top = r.as_euler('xyz', degrees=False)
    rot = copy.deepcopy(reg_plane.transformation[:3,:3])
    r = Rotation.from_matrix(rot)
    euler_ang_plane = r.as_euler('xyz', degrees=False)

    dx = reg_top.transformation[0,3]
    dy = reg_top.transformation[1,3]
    dz = reg_plane.transformation[2,3]
    translation = np.array([dx, dy, dz])
    euler_angs = np.array([euler_ang_plane[0], euler_ang_plane[1], euler_ang_top[2]])

    new_rot = Rotation.from_euler('xyz', euler_angs).as_matrix()
    transformation = np.c_[new_rot, translation]
    transformation = np.r_[transformation, np.array([0,0,0,1], ndmin=2)]
    
    if return_info:
        information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(target, source, threshold, transformation)
        evaluation = o3d.pipelines.registration.evaluate_registration(target, source, threshold, transformation )
        return transformation, information_matrix, evaluation
    return transformation




# if __name__ == "__main__":


    # pcd = load_pcd_from_index(400, path_to_pcds).voxel_down_sample(0.01)
    # location_histogram = create_location_histogram(pcd, density=0.1, area=10)
    # plt.figure()
    # plt.imshow(location_histogram, cmap="bw")
    # plt.show()








    # pcd2 = load_pcd_from_index(999+1, path_to_pcds).voxel_down_sample(0.01)
    # ground_plane_pcd1,_,_, model1 = extract_ground_plane(pcd1, 0.1, max_iteration=100)
    # ground_plane_pcd2,_,_, model2 = extract_ground_plane(pcd2, 0.1, max_iteration=100)
    # print(model1)
    # fitting_plane1 = construct_plane_from_model(model1,size=20, density=10)
    # fitting_plane2 = construct_plane_from_model(model2,size=20, density=10)
    # fitting_plane = construct_plane_from_model(size=30, density=10)
    # init = np.eye(4)
    # init[1,3] = 1.0
    # reg0 = estimate_p2pl(fitting_plane1, fitting_plane2, use_coarse_estimate=False)
    # print(reg0.transformation)
    # rot = copy.deepcopy(reg0.transformation[:3,:3])
    # r = Rotation.from_matrix(rot)
    # print(r.as_euler('xyz', degrees=True))
    # print(np.rad2deg(rotation_matrix2euler_angles(rot)))
    # reg1 = estimate_p2pl(ground_plane_pcd1, ground_plane_pcd2, use_coarse_estimate=True)
    # print(reg1.transformation)
    # reg2, info_, eva_ = estimate_p2pl(pcd1, pcd2, use_coarse_estimate=True, return_info=True)
    # print(reg2.transformation, eva_, sep='\n')
    # transformation_GPC, info, eva = ground_plane_constrained_ICP(pcd1, pcd2, return_info=True, threshold=0.2)
    # print(transformation_GPC, eva, sep='\n')
    # rot = copy.deepcopy(reg3.transformation[:3,:3])
    # r = Rotation.from_matrix(rot)
    # print(r.as_euler('xyz', degrees=True))
    # print(reg)
    # draw_pcd([ground_plane_pcd1, fitting_plane])
    # draw_pcd([
    #     fitting_plane1.paint_uniform_color([0.0,0.0,1.0]),
    #     fitting_plane2.paint_uniform_color([1.0,0.0,0.0])])
    # draw_registration_result(fitting_plane1, fitting_plane2, reg0.transformation)
    # draw_registration_result(ground_plane_pcd1, ground_plane_pcd2, reg1.transformation)
    # draw_registration_result(pcd1, pcd2, reg2.transformation)
    # draw_registration_result(pcd1, pcd2, transformation_GPC)
    # draw_registration_result(ground_plane_pcd1, ground_plane_pcd2, reg.transformation)