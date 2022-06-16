import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import pickle
# import copy
# import os
# import sys
# import pyquaternion
# import time
# import concurrent.futures
# import progress_bar
# import colorsys
# import threading

# from undistort_pcd_test import *
# from functions import *
import plotting
from load_paths import * # loads paths to data


# def plot_odometry(odometry):
#     fig, ax = plt.subplots(2)
#     ax[0].plot(odometry[:,0], odometry[:,1])
#     ax[1].plot(odometry[:,0], odometry[:,2])
def get_processed_folder(run):
    folders = listdirs(PATH_TO_PROCESSED)
    for f in folders:
        if int(f.split('_')[-1]) == run:
            return f
    print(f'no run with index {run}')

def load_processed_odometry(run):
    folder = get_processed_folder(run)
    return np.load(f'{PATH_TO_PROCESSED}/{folder}/odometry.npy')

def load_odometry_and_orientation(run):
    folder = get_processed_folder(run)
    # try:
    pose_data_file = open(f'{PATH_TO_PROCESSED}/{folder}/poses.txt','r')
    #     except:
    # pose_data_file = open(f'{PATH_TO_PROCESSED}/{folder}/poses.csv','r')
        
    pose_data = pose_data_file.readlines()
    position = []
    orientation = []
    for pose in pose_data:
        pose_matrix = np.array([float(s) for s in  str.split(pose)])
        pose_matrix = pose_matrix.reshape((3,4))
        pos = pose_matrix[:,3]
        orien = pose_matrix[:3,:3]
        position.append([pos[0], pos[1], pos[2]])
        orientation.append(orien)
    pose_data_file.close()
    return np.asarray(position), np.asarray(orientation)

def load_processed_map(run):
    folder = get_processed_folder(run)
    return o3d.io.read_point_cloud(f'{PATH_TO_PROCESSED}/{folder}/SLAM_map.pcd')

def load_settings(run):
    folder = get_processed_folder(run)
    try:
        with open(f'{PATH_TO_PROCESSED}/{folder}/settings.pickle', 'rb') as handle:
            settings = pickle.load(handle)
        print(settings)
        return settings
    except:
        print('No settings.pickle to read.. :/')


def estimate_alignment(p,q, yaw_only=False):
    """
    This functions estimates the alignments parameteres s, R, T, described in "A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry"
    This is for better evaluation of the odometry error, described in the same paper. The method is shown in algorithm 1.
    """
    # yaw alignment only
    if yaw_only:
        p = np.atleast_2d(p[:,:2])
        q = np.atleast_2d(q[:,:2])

    #dimension
    try:
        dim = p.shape[1]
    except:
        dim = len(p)

    # Calculate the centroids 
    centroid_p = np.mean(p, axis = 0, keepdims=True)
    centroid_q = np.mean(q, axis = 0, keepdims=True)

    # subtract mean
    P = p - centroid_p
    Q = q - centroid_q

    
    # find scale
    # find lengths and sum
    # P_norm = np.linalg.norm(P,axis=1)
    # Q_norm = np.linalg.norm(Q,axis=1)
    
    # P_sum = np.sum(P_norm)
    # Q_sum = np.sum(Q_norm)
    #s = Q_sum/P_sum
    s = 1 # scale is known and should be 1


    # find the rotation
    C = Q.T @ P
    
    U, S, V_t = np.linalg.svd(C)
    R = U @ V_t
    det_R = np.linalg.det(R)
    # D = np.array([[1,0,0],[0,1,0],[0,0, det_R]])
    D = np.eye(dim)
    D[dim-1,dim-1] = det_R
    R = R @ D

    

    # find translation
    t = centroid_q.T - s * R @ centroid_p.T

    if R.shape[0] == 2:
        _R = np.eye(3)
        _R[:2,:2] = R
        R = _R
        t = t.ravel()
        t = np.atleast_2d(np.r_[t, 0.0]).T

    return s, R, t

def align_odometry(p, s_p,R_p,t_p):
    aligned_P = s_p* R_p@p.T + t_p
    return aligned_P.T

def align_orientation(R, R_p):
    return R_p.T @ R

def calc_rmse(residual):
    rmse_x = np.sqrt(np.mean(residual[:,0]**2))
    rmse_y = np.sqrt(np.mean(residual[:,1]**2))
    rmse_z = np.sqrt(np.mean(residual[:,2]**2))
    norm = np.linalg.norm(residual, axis=1)
    rmse_norm = np.sqrt(np.mean( norm**2 ))

    rmse = dict(x = rmse_x, y=rmse_y, z=rmse_z, norm = rmse_norm)
    return rmse
    

def calc_L2(residual): 
    L2_error_x = np.sqrt(residual[:,0].T @ residual[:,0]) 
    L2_error_y = np.sqrt(residual[:,1].T @ residual[:,1])
    L2_error_z = np.sqrt(residual[:,2].T @ residual[:,2])
    residual_abs = np.linalg.norm(residual, axis=1)
    L2_error_abs = np.sqrt(residual_abs.T @ residual_abs)

    L2 = dict(x = L2_error_x, y=L2_error_y, z=L2_error_z, norm = L2_error_abs)
    return L2

def calc_delta_R(R_truth, R_aligned):
    kitti_rot_mat = np.array([[0,-1,0],[0,0,-1],[-1,0,0]])
    R_truth_euler = (kitti_rot_mat @ Rotation.from_matrix(R_truth).as_euler('yzx', degrees=False).T).T
    R_aligned_euler = Rotation.from_matrix(R_aligned).as_euler('xyz', degrees=False)
    # R_truth = Rotation.from_euler('xyz',(kitti_rot_mat.T@ R_truth_euler.T).T ).as_matrix()
    delta_R_euler = R_truth_euler - R_aligned_euler
    delta_R_euler[delta_R_euler>np.pi/2] -= 2*np.pi
    delta_R_euler[delta_R_euler<-np.pi/2] += 2*np.pi
    delta_R = Rotation.from_euler('xyz', delta_R_euler, degrees=False).as_matrix()
    return delta_R

def calc_ATE(p_truth, R_truth, p_aligned, R_aligned):
    # print(R_truth[0])
    # print(R_aligned[0])
    # kitti_rot_mat = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
    # kitti_rot_mat = np.array([[0,-1,0],[0,0,-1],[-1,0,0]])
    # R_truth_euler = (kitti_rot_mat @ Rotation.from_matrix(R_truth).as_euler('yzx', degrees=False).T).T
    # R_aligned_euler = Rotation.from_matrix(R_aligned).as_euler('xyz', degrees=False)
    # # R_truth = Rotation.from_euler('xyz',(kitti_rot_mat.T@ R_truth_euler.T).T ).as_matrix()
    # delta_R_euler = R_truth_euler - R_aligned_euler
    # delta_R_euler[delta_R_euler>np.pi/2] -= 2*np.pi
    # delta_R_euler[delta_R_euler<-np.pi/2] += 2*np.pi
    # delta_R = Rotation.from_euler('xyz', delta_R_euler, degrees=False).as_matrix()
    delta_R = calc_delta_R(R_truth, R_aligned)
    # delta_R = R_truth @ R_aligned.transpose((0,2,1))
    # test = (delta_R @ p).reshape((len(p_aligned),3))
    p = p_aligned.reshape((len(p_aligned),3,1))
    delta_p = (p_truth - (delta_R @ p).reshape((len(p_aligned),3)))
    ate_x = np.sqrt(np.mean(delta_p[:,0]**2))
    ate_y = np.sqrt(np.mean(delta_p[:,1]**2))
    ate_z = np.sqrt(np.mean(delta_p[:,2]**2))
    ate_pos = np.sqrt(np.mean(np.linalg.norm(delta_p, axis=1)**2))

    delta_euler = Rotation.from_matrix(delta_R).as_euler('xyz', degrees=True)
    delta_R_deg = np.linalg.norm(Rotation.from_matrix(delta_R).as_rotvec(degrees=True), axis=1)
    ate_rot_x = np.sqrt(np.mean(delta_euler[:,2]**2))
    ate_rot_y = np.sqrt(np.mean(delta_euler[:,1]**2))
    ate_rot_z = np.sqrt(np.mean(delta_euler[:,0]**2))
    ate_rot = np.sqrt(np.mean(delta_R_deg**2))

    ATE_pos =  dict(x = ate_x, y=ate_y, z=ate_z, norm = ate_pos)
    ATE_rot =  dict(x = ate_rot_x, y=ate_rot_y, z=ate_rot_z, norm = ate_rot)


    delta_R_deg = np.linalg.norm(Rotation.from_matrix(delta_R).as_rotvec(degrees=True), axis=1)
    R_truth_deg = np.linalg.norm(Rotation.from_matrix(R_truth).as_rotvec(degrees=True), axis=1)
    R_aligned_deg = np.linalg.norm(Rotation.from_matrix(R_aligned).as_rotvec(degrees=True), axis=1)
    

    # R_aligned_euler = Rotation.from_rotvec(R_aligned_euler).as_mrp()
    # R_truth_euler = Rotation.from_rotvec(R_truth_euler).as_mrp()
    # plt.plot(delta_R_deg)
    # plt.plot(R_truth_deg)
    # plt.plot(R_aligned_deg)
    # plt.plot(R_aligned_euler,ls = '--', label=['x', 'y', 'z'])
    # plt.plot( R_truth_euler, label=['x', 'y', 'z'])
    # plt.plot( R_truth_euler-R_aligned_euler, label=['y', 'z', 'x'])
    # plt.plot(delta_R_euler, label=['x', 'y', 'z'])
    # plt.legend()
    # plt.show()

    return ATE_pos, ATE_rot


def calc_motion_parameters(odometry):
    step_length = np.diff(odometry, axis=0)
    step_length_norm = np.linalg.norm(step_length, axis=1 )
    cumulative_step_length_xyz = np.r_[np.array(np.zeros((1,3))),np.cumsum(abs(step_length), axis=0)]
    cumulative_step_length = np.r_[0.0, np.cumsum(step_length_norm)]
    travelled_distance_xyz = np.sum(abs(step_length), axis=0)
    # travelled_distance = np.linalg.norm(travelled_distance_xyz, axis=0)
    travelled_distance = np.sum(np.linalg.norm(step_length,axis=1),axis=0)

    # velocity = step_length / delta_t
    # speed = np.linalg.norm(velocity, axis=1)

    return cumulative_step_length, travelled_distance, travelled_distance_xyz

def calc_relative_error(odometry, R_aligned, odometry_truth, R_truth, dist=[10]):
    if not isinstance(dist, list):
        dist = [dist]
    delta_R = calc_delta_R(R_truth, R_aligned)
    cum_length = calc_motion_parameters(odometry)[0]
    RE = dict(delta_p=[], delta_phi=[], segment_length=[])
    for d in dist:
        segment_length = d
        e = []
        s = []
        for i in range(len(cum_length)):
            for j in range(i+1,len(cum_length)):
                if cum_length[j] - cum_length[i] > segment_length:
                    s.append(i)
                    e.append(j)
                    break

        delta_p = []
        delta_phi = []
        for k, (_s,_e) in enumerate(zip(s,e)):
            d_ = [odometry[_s], odometry[_e]]
            d_truth = [odometry_truth[_s], odometry_truth[_e]]
            # alignment_params = estimate_alignment(d[0],d_truth[0])
            aligned_d = align_odometry(np.asarray(d_), s_p=1, R_p=delta_R[_s],t_p=np.array( d_truth[0] - delta_R[_s]@d_[0],ndmin=2).T)
            test = aligned_d - d_truth
            d_R_k = (R_aligned[_s].T @ R_aligned[_e]).T @ R_truth[_s].T @ R_truth[_e]
            d_phi = np.linalg.norm(Rotation.from_matrix(d_R_k).as_rotvec(degrees=True))
            # d_phi = np.linalg.norm(Rotation.from_matrix(delta_R[_e]).as_rotvec(degrees=True))
            # error_aligned_d = (np.array(d_truth).T - delta_R[_e] @ np.array(aligned_d).T).T
            error_aligned_d = (d_R_k @ np.array(aligned_d).T).T 
            error_aligned_d_old = (delta_R[_e] @ np.array(aligned_d).T).T 
            error_aligned_d = error_aligned_d - (error_aligned_d[0,:] - d_truth[0] )
            d_p = np.linalg.norm(d_truth[1] - error_aligned_d[1,:])
            # d_p_test = np.linalg.norm(d_truth[-1] - delta_R[_e].T @ aligned_d[-1])
            # plt.clf()
            # plot_segment(d_,label='d_', c='C0')
            # plot_segment(d_truth,label='d_truth', c='C1')
            # plot_segment(aligned_d,label='aligned_d', c='C2')
            # plot_segment(error_aligned_d,label='error_aligned_d', c='C3')

            # plt.legend()

            delta_p.append(d_p)
            delta_phi.append(d_phi)

        RE['delta_p'].append(np.array(delta_p))
        RE['delta_phi'].append(np.array(delta_phi))
        RE['segment_length'].append(d)


    return RE

def plot_segment(d, label='', c='C0'):
    d = np.atleast_2d(np.asarray(d))
    plt.plot(d[:,0],d[:,1], label=label, color=c)

def plot_relative_error(RE, odometry):
    cum_length = calc_motion_parameters(odometry)[0]
    fig1, (ax1, ax2) = plt.subplots(2, 1,constrained_layout=True)
    for i, d in enumerate(RE['segment_length']):
        ax1.plot(cum_length[:len(RE['delta_p'][i])],RE['delta_p'][i], label=d)
        ax2.plot(cum_length[:len(RE['delta_phi'][i])],RE['delta_phi'][i],label=d)
    ax1.legend()
    ax1.set_ylabel('Relative Translational error [m]')
    ax2.set_ylabel('Relative rotational error [deg]')
    ax2.set_xlabel('Travelled Distance [m]')

    fig2, (ax1, ax2) = plt.subplots(2, 1,constrained_layout=True,sharex=True)
    for i, d in enumerate(RE['segment_length']):
        ax1.scatter(np.random.normal(i+1,0.05,len(RE['delta_p'][i])), RE['delta_p'][i],marker='.', s=0.5,c='C0', alpha=0.5)
        ax2.scatter(np.random.normal(i+1,0.05,len(RE['delta_phi'][i])), RE['delta_phi'][i],marker='.', s=0.5,c='C0', alpha=0.5)
    
    ax1.boxplot(RE['delta_p'],showfliers = False)
    ax1.set_ylabel('Segment translational Error [m]')
    
    ax2.boxplot(RE['delta_phi'],showfliers = False)
    ax2.set_ylabel('Segment rotational Error [deg]')
    
    ax2.set_xticks(range(1,len(RE['segment_length'])+1))
    ax2.set_xticklabels(RE['segment_length'])
    ax2.set_xlabel('Segment Distance [m]')

    fig2.suptitle(f'Relative Error of {DATA_SET_LABEL}, Loop Closure: {USE_LOOP_CLOSURE}')
    #plot statistical  box-plot ish
    # odometry colored with the error to see where the error low and high

    fig3, (ax1, ax2) = plt.subplots(1, 2 , constrained_layout=True)
    plotting.plot_odometry_colorbar_2D(odometry, c_values= RE['delta_p'][0], fig=fig3, ax=ax1, cmap='plasma')
    plotting.plot_odometry_colorbar_2D(odometry, c_values= RE['delta_phi'][0], fig=fig3, ax=ax2, cmap='plasma')
    
    
    return



def evaluate_odometry_to_ground_thruth(odometry, orientations, truth_odometry, truth_orientations, alignment=True):
    if len(odometry) -1 == len(truth_odometry):
        odometry = odometry[:-1,:]
    # truth_odometry,truth_orientations = get_closest_truth_value(odometry, truth_odometry, truth_orientations) # dårlig ide
    if alignment:
        alignment_params = estimate_alignment(odometry, truth_odometry)
        # alignment_params_yaw = estimate_alignment(odometry, truth_odometry, yaw_only=True)
        aligned_odometry = align_odometry(odometry, *alignment_params)
        # yaw_aligned_odometry = align_odometry(odometry, *alignment_params_yaw)
        aligned_orientations = align_orientation(orientations, alignment_params[1])
        # yaw_aligned_orientations = align_orientation(orientations, alignment_params_yaw[1])
    else:
        aligned_odometry = odometry
        aligned_orientations = orientations

    # yaw_aligned_residual = yaw_aligned_odometry - truth_odometry
    residual = aligned_odometry - truth_odometry
    residual_non_aligned = odometry - truth_odometry
    residual_norm = np.linalg.norm(residual, axis=1)
    residual_non_aligned_norm = np.linalg.norm(residual_non_aligned, axis=1)
    
    cumulative_step_length, travelled_distance, travelled_distance_xyz = calc_motion_parameters(odometry)

    relative_deviation = np.divide(residual_norm, cumulative_step_length, out=np.zeros_like(residual_norm), where=(cumulative_step_length > 0)) * 100 # this is not correct
    endpoint_relative_deviation_xyz = residual_non_aligned[-1]/travelled_distance_xyz * 100 # in percent
    endpoint_relative_deviation = residual_non_aligned_norm[-1]/travelled_distance * 100 # in percent
    end_point_relative_dev = np.r_[endpoint_relative_deviation_xyz, endpoint_relative_deviation]
    end_point_dev = np.r_[residual_non_aligned[-1], residual_non_aligned_norm[-1]]

    # RE = calc_relative_error(aligned_odometry,aligned_orientations, truth_odometry, truth_orientations,d=10)
    RE = calc_relative_error(odometry, orientations, truth_odometry, truth_orientations, dist=[2,5,10,15,25])
    ATE_pos, ATE_rot = calc_ATE(truth_odometry, truth_orientations, aligned_odometry, aligned_orientations)
    # ate_yaw_pos, ate_yaw_rot = calc_ATE(truth_odometry, truth_orientations, yaw_aligned_odometry, yaw_aligned_orientations)

    L2 = calc_L2(residual_non_aligned) 
    rmse = calc_rmse(residual)
    travelled_distance_xyz_norm = np.r_[travelled_distance_xyz, travelled_distance]

    labels = ['x','y','z', 'norm']
    df = pd.DataFrame({
                       'ATE pos [m]':ATE_pos,
                       'ATE rot [deg]':ATE_rot,
                    #    'ATE pos (yaw) [m]': ate_yaw_pos,
                    #    'ATE rot (yaw) [m]': ate_yaw_rot,
                       'L2 [m]': L2,
                       'RMSE [m]':rmse, 
                       'endpoint dev. [m]': end_point_dev,
                       'endpoint rel. dev. [%]': end_point_relative_dev,
                       'total distance [m]':travelled_distance_xyz_norm
                       }, 
                        index=labels)
    print(df)

    plot_relative_error(RE, aligned_odometry)
    
    

    fig, (ax1, ax2) = plt.subplots(2, 1,constrained_layout=True)
    ax1.plot(cumulative_step_length, residual[:,0], label='X')
    ax1.plot(cumulative_step_length, residual[:,1], label='Y')
    ax1.plot(cumulative_step_length, residual[:,2], label='Z')
    ax1.plot(cumulative_step_length,  residual_norm, label='±Norm', ls='--', lw=0.7, color='r')
    ax1.plot(cumulative_step_length, -residual_norm, ls='--', lw=0.7, color='r')
    ax1.axhline(y=0, linestyle='--', color='k', alpha =0.5, lw=0.7)

    

    ax1.set_ylabel('Positional Error [m]')
    ax1.set_xlabel('Travelled Distance [m]')
    ax1.legend()

    ax2.plot(cumulative_step_length, odometry[:,0], label='Odometry, X', color='C0')
    ax2.plot(cumulative_step_length, odometry[:,1], label='Odometry, Y', color='C1')
    ax2.plot(cumulative_step_length, odometry[:,2], label='Odometry, Z', color='C2')

    ax2.plot(cumulative_step_length, truth_odometry[:,0], label='Ground Truth, X', color='C0', linestyle='--')
    ax2.plot(cumulative_step_length, truth_odometry[:,1], label='Ground Truth, Y', color='C1', linestyle='--')
    ax2.plot(cumulative_step_length, truth_odometry[:,2], label='Ground Truth, Z', color='C2', linestyle='--')

    ax2.set_ylabel('Odometry position [m]')
    ax2.set_xlabel('Travelled Distance [m]')
    ax2.legend()

    fig.suptitle(f'{DATA_SET_LABEL} - Sequence {SEQ_LABEL}')


    # Odometry plot
    fig,axes = plotting.plot_odometry_2D(truth_odometry, label='Ground Truth', color='k', lw=3.0)
    # plotting.plot_odometry_2D(odometry, label='Odometry', axes=axes)
    plotting.plot_odometry_2D(aligned_odometry, label='Aligned Odometry', axes=axes)
    # plotting.plot_odometry_2D(yaw_aligned_odometry, label='Yaw Aligned Odometry', axes=axes)
    fig.suptitle(f'{DATA_SET_LABEL} - Sequence {SEQ_LABEL}')

def get_closest_truth_value(odometry, truth, truth_orientations):
    neigh = NearestNeighbors(n_neighbors=10, radius=50)
    neigh.fit(truth)
    closest_thruth_idx = neigh.kneighbors(odometry, return_distance=False)
    closest_thruth_idx = np.min(closest_thruth_idx, axis=1)
    reduced_truth = truth[closest_thruth_idx,:].reshape((-1,3))
    reduced_truth_orientations = truth_orientations[closest_thruth_idx,:]
    return reduced_truth, reduced_truth_orientations


if __name__ == "__main__":
    RUN = 13

    # odometry = np.load(r'/Volumes/HAMMER DATA 2TB/DTU_LIDAR_20220523/20220523_122236/processed_data/20220523_122236_run_001/odometry.npy')
    load_settings(RUN)
    odometry = load_odometry_and_orientation(RUN)
    if path_to_pose_txt.split('.')[-1] == 'csv':
        n = len(odometry[0])
        ground_truth = plotting.get_pose_ground_truth_DTU(path_to_pose_txt, n=n)

    else:
        ground_truth = plotting.get_pose_ground_truth_KITTI(path_to_pose_txt)



    # _,axes = plotting.plot_odometry_2D(ground_truth[0], marker='.', label='Ground Truth')
    # plotting.plot_odometry_2D(odometry[0],axes=axes, marker='.')


    evaluate_odometry_to_ground_thruth(*odometry, *ground_truth, alignment=False)



    plt.show()


    # slam_map = load_processed_map(0)#.voxel_down_sample(0.5)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.get_render_option().load_from_json(path_to_cwd + 'render_options_vis.json')
    # vis.add_geometry(slam_map)

    # while True:
    #     if not vis.poll_events():
    #         break
    #     vis.update_renderer()
