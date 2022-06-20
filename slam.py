import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys
import pyquaternion
import time
# import concurrent.futures
import progress_bar
import colorsys
# import threading
import pandas as pd

# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise

from collections import Counter

from undistort_pcd_test import *
from functions import *
from load_paths import * # loads paths to data


# class KeyboardThread(threading.Thread):

#     def __init__(self, input_cbk = None, name='keyboard-input-thread', mesg=''):
#         self.input_cbk = input_cbk
#         self.mesg = mesg
#         super(KeyboardThread, self).__init__(name=name)
#         self.start()
#         self.ret = None

#     def run(self):
#         # while True:
#         self.ret = self.input_cbk(input(self.mesg)) #waits to get input + Return

#     def retrieve(self):
#         return self.ret





class Keyframe(object):
    def __init__(self, id, pcd, pcd_full, odometry, orientation, frame_idx):
        self.id = id                         # the corresponding frame index
        self.frame_idx = frame_idx
        self.odometry = copy.deepcopy(odometry)      # the odometry pose
        self.pcd = copy.deepcopy(pcd) # downscaled pcd
        self.pcd_full = copy.deepcopy(pcd_full)       # full point cloud data
        self.slam_transformed_pcd = copy.deepcopy(pcd)
        self.slam_transformed_pcd_full = copy.deepcopy(pcd_full)
        self.slam_transformed_pcd.paint_uniform_color(colorsys.hsv_to_rgb(self.id/256, 1.0, 1.0))
        self.slam_transformed_pcd_full.paint_uniform_color(colorsys.hsv_to_rgb(self.id/256, 1.0, 1.0))
        self.position = self.odometry[:3,3]
        self.orientation = np.eye(3) #temp override of input: orientation
        # self.last_vis_update = 0

        self.pose = self.odometry
        # self.pose_graph_node = o3d.pipelines.registration.PoseGraphNode(odometry)
        # pose_graph.nodes.append(self.pose_graph_node)

        self.orientation_LineSet = o3d.geometry.LineSet()
        # self.update_coordinate_frame()
        self.orientation_LineSet.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
        self.orientation_LineSet.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # self.update_slam_transform(pose_graph)

    # updates the keyframes  after optimization
    def update_slam_transform(self, pose, full=False):
        # self.odometry = self.pose_graph_node.pose
        # "reset" the transform
        self.slam_transformed_pcd.points = self.pcd.points
        self.slam_transformed_pcd.normals = self.pcd.normals
        # self.slam_transformed_pcd = copy.deepcopy(self.pcd) # then i becomes a new one, and 
        # redo the transform after optimization
        # self.slam_transformed_pcd.transform(self.pose_graph_node.pose)
        self.pose = copy.deepcopy(pose)
        self.slam_transformed_pcd.transform(self.pose)
        if full:
            self.slam_transformed_pcd_full.points = self.pcd_full.points
            self.slam_transformed_pcd_full.normals = self.pcd_full.normals
            self.slam_transformed_pcd_full.transform(self.pose)

        self.update_coordinate_frame()



    def update_coordinate_frame(self):
        self.coord_points = []
        self.position = self.pose[:3,3]
        self.coord_points.append(self.position) # center pos of the rgb orientation axes
        for row in (self.pose[:3,:3]@ self.orientation).T:
            self.coord_points.append( row + self.position)
        self.orientation_LineSet.points = o3d.utility.Vector3dVector(self.coord_points)
        

class Odometry(object):
    # use bundle adjustment from keyframe optimisation
    def __init__(self):
        self.odometry = []
        self.non_opt_odometry = []
        self.positions = []
        self.orientations = []
        self.lines = []
        self.LineSet = o3d.geometry.LineSet()
        self.LineSet.lines = o3d.utility.Vector2iVector([[0,0]])

        # self.init_plot()

        # self.coordinate_frame = [[0,0,0],[0, 0, 0], [1, 0, 0], [0, 0, 1]]
        # self.coordinate_frame_lines = [[0, 1], [0, 2], [0, 3]]
        # self.coordinate_frame_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def init_plot(self):
        self.fig, self.axes = plt.subplots(1,3, figsize=(12,4)) 
        self.odometry_plot = self.axes[0].plot(0,0)[0]
        self.axes[0].set_xlabel("X [m]")
        self.axes[0].set_ylabel("Y [m]")
        self.elevationX_plot = self.axes[1].plot(0,0)[0]
        self.axes[1].set_xlabel("X [m]")
        self.axes[1].set_ylabel("Z [m]")
        self.elevationY_plot = self.axes[2].plot(0,0)[0]
        self.axes[2].set_xlabel("Y [m]")
        self.axes[2].set_ylabel("Z [m]")
        self.fig.tight_layout()

    def push_odometry(self, pose, non_opt_pose, init=False):
        self.odometry.append(pose)
        self.positions.append(pose[:3,3])
        self.orientations.append(pose[:3,:3])

        self.non_opt_odometry.append(non_opt_pose)
        
        # if len(self.odometry) > 1: #not init: 
        # self.push_lineset()
            # self.update_plot()

    def push_lineset(self): # denne funktion laver segmentation fault!
        self.LineSet.points = o3d.utility.Vector3dVector(self.positions)
        i = len(self.LineSet.points)
        if i > 1:
            self.lines.append([i-2, i-1])
            self.LineSet.lines = o3d.utility.Vector2iVector(self.lines)

        self.LineSet.paint_uniform_color([1.0,1.0,1.0])

    def update_odometry(self, nodes):
        for i, node in enumerate(nodes):
            self.odometry[i] = node.pose
            self.positions[i] = node.pose[:3,3]
            self.orientations[i] = node.pose[:3,:3]
        
        self.update_lineset()


    def update_lineset(self): # redo the lineset
        # pull the the updated positions
        self.LineSet.points = o3d.utility.Vector3dVector(self.positions)
        # self.lines.append([len(self.odometry_LineSet.points)-1 , len(self.odometry_LineSet.points)])
        self.lines = [[i,i+1] for i in range( len(self.LineSet.points)-1) ] # append? instead of recreate all line segments
            
        # print(self.lines)
        self.LineSet.lines = o3d.utility.Vector2iVector(self.lines)
        self.LineSet.paint_uniform_color([1.0,1.0,1.0])


    def update_plot(self):
        # print(self.positions)
        pos = np.asarray(self.positions)
        # print(pos)
        self.odometry_plot.set_data(pos[:,0], pos[:,1])
        self.elevationX_plot.set_data(pos[:,0], pos[:,2])
        self.elevationY_plot.set_data(pos[:,1], pos[:,2])
        for ax in self.axes:
            # ax.axis("square")
            # ax.set_box_aspect(1)
            ax.relim()
            ax.autoscale_view(True,True,True)
            ax.set_aspect('equal', adjustable='box')
            # ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        self.fig.canvas.draw_idle()
        # self.fig.canvas.draw()
        plt.pause(0.000001)
        # self.fig.canvas.show()
        # self.fig.canvas.flush_events()


class Loop_Closure(object):
    def __init__(self):
        self.lineset = o3d.geometry.LineSet()
        self.lines = []
        self.positions = []
        self.kf_index = []

        self.lineset.points = o3d.utility.Vector3dVector(self.positions)
        self.lineset.lines = o3d.utility.Vector2iVector( self.lines)

        # self.lineset.paint_uniform_color([1.0,0.0,1.0])
        
    def push_loop_closure(self, keyframe1, keyframe2):
        self.kf_index.append([keyframe1.id, keyframe2.id])
        self.positions.append(keyframe1.position)
        self.positions.append(keyframe2.position)

        idx = len(self.positions)-1
        self.lines.append([idx-1, idx]) # the index of which keyframe positions are connected by a line

        self.lineset.points = o3d.utility.Vector3dVector(self.positions)
        self.lineset.lines = o3d.utility.Vector2iVector(self.lines)

        self.lineset.paint_uniform_color([1.0,0.0,1.0]) # paint the loop closure lines purple
    
    def update_lineset(self, keyframes):
        self.positions = []
        for idx1, idx2 in self.kf_index: # do this without a loop?
            self.positions.append(keyframes[idx1].position)
            self.positions.append(keyframes[idx2].position)

        self.lineset.points = o3d.utility.Vector3dVector(self.positions)
        self.lineset.lines = o3d.utility.Vector2iVector(self.lines)

        # self.lineset.paint_uniform_color([1.0,0.0,1.0])


# visualizer class?
    
# class KalmanFilter(object):
#     def __init__(self):
#         self.dt = 0.2
#         self.state = np.zeros((6,1))
#         self.state
#         self.prediction = np.zeros_like(self.state)


#         #state transition matrix
#         self.F = np.eye(6)
#         self.F[3:,3:] = np.eye(3)#*self.dt

#         # state covariance
#         self.P = np.diag([1,1,1,0.5,0.5,0.5])

#         # process noise
#         self.Q

#         self.H = np.diag([1,1,1,0,0,0])



#     def predict(self):
#         self.prediction = self.F @ self.state + np.eye(6) @ self.P


class PG_SLAM(object):
    def __init__(self, bias_correction, **params):    #calib_lidar_pose=None, undistort_pcd=False, cam_follow=True, use_loop_closure=True):
        self.finish = False
        self.pose_graph = o3d.pipelines.registration.PoseGraph() # it has the pose graph
        self.cycle = 0
        self.last_vis_update = 0
        self.undistort_pcd = params['undistort_pcd']
        self.cam_follow = params['cam_follow']
        self.use_loop_closure = params['use_loop_closure']
        self.use_local_map_icp = params['use_local_map_icp']

        # class handling the odometry graphics, but needs bundle adjustment to work with the pose graph optimization
        self.Odometry = Odometry()
        # self.Odometry.update_odometry(np.eye(4))

        # class handling loop closures graphics
        self.Loop_Closure = Loop_Closure()

        self.translation_bias = np.array([bias_correction["x"], bias_correction["y"], bias_correction["z"]], ndmin=1)

        z_rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0.0,0.0,1.0]) * np.deg2rad(bias_correction["yaw"]))
        self.z_rot_bias = np.eye(4)
        self.z_rot_bias[:3,:3] = z_rot_mat
        
        roll_rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0.0,1.0,0.0]) * np.deg2rad(bias_correction["roll"]))
        self.roll_rot_bias = np.eye(4)
        self.roll_rot_bias[:3,:3] = roll_rot_mat

        pitch_rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0,0.0,0.0]) * np.deg2rad(bias_correction["pitch"]))
        self.pitch_rot_bias = np.eye(4)
        self.pitch_rot_bias[:3,:3] = pitch_rot_mat

        if params['calib_lidar_pose'] is None:
            # self.lidar_pose = self.z_rot_bias
            self.lidar_pose = np.eye(4)
        else:
            self.lidar_pose = params['calib_lidar_pose']

        r = Rotation.from_matrix(self.lidar_pose[:3,:3])
        self.euler_ang_lidar = r.as_euler('xyz', degrees=True)
        print(f'Eulers angles of lidar "XYZ": {self.euler_ang_lidar}')



        self.process_times = []
        self.icp_times = []
        self.fitnesses = []
        self.inlier_rmses = []
        self.proximites_count = 0



        self.recent_proximity_ids = []

        self.keyframes = [] # an empty list for keyframes
        self.keyframe_table = [] # table to index frame to keyframe id
        self.last_transform = np.eye(4) # the initial pose/transformation
        self.last_transform_guess = np.eye(4) # the initial pose/transformation "guess"
        self.last_odometry_transform = np.eye(4) # the incremental steps of each frame
        self.previous_odometry_transform = np.eye(4) # the incremental steps of each frame
        self.timestamps = []
        self.delta_time = 0.0

        # self.last_odometry = self.lidar_pose # start with the orientation of the lidar
        self.last_odometry = np.eye(4)
        self.odometry_not_opt = self.last_odometry

        self.velocity = np.zeros((3,1))
        self.position = np.zeros((3,1))
        self.gps_position = np.zeros((3,1))


        # self.prev_translation = translation_dist_of_transform(self.last_transform)
        # self.prev_rotation = angular_rotation_of_transform(self.last_transform)

        # some settings
        self.keyframe_angle_threshold = 15.0 # in degres
        self.keyframe_translation_threshold = 5.0 # in meters
        self.ICP_treshold = 0.5

        self.optimization_options = o3d.pipelines.registration.GlobalOptimizationOption(
                                        max_correspondence_distance=0.5,
                                        edge_prune_threshold=0.25,
                                        preference_loop_closure=200.0,
                                        reference_node=0)


        # GRAPHICS
        # visulazier for loop closure control
        self.vis_loop = o3d.visualization.Visualizer()
        self.vis_loop.create_window()
        self.vis_loop.get_render_option().load_from_json(path_to_cwd + 'render_options_vis.json')
        point = np.random.normal(0.0,20.0,(100,3))
        point_pcd = o3d.geometry.PointCloud()
        point_pcd.points = o3d.utility.Vector3dVector(point)
        self.pcd1 = copy.deepcopy(point_pcd)
        self.pcd2 = copy.deepcopy(point_pcd)
        self.vis_loop.add_geometry(self.pcd1)
        self.vis_loop.add_geometry(self.pcd2)
        # self.vis_loop.update_renderer()

        # vis for current pcd
        self.vis_pcd = o3d.visualization.Visualizer()
        self.vis_pcd.create_window()
        self.vis_pcd.get_render_option().load_from_json(path_to_cwd + 'render_options_vis.json')
        self.pcd_vis = copy.deepcopy(point_pcd)
        self.vis_pcd.add_geometry(self.pcd_vis)



        # visulazier for loop ICP
        self.vis_ICP = o3d.visualization.Visualizer()
        self.vis_ICP.create_window()
        self.vis_ICP.get_render_option().load_from_json(path_to_cwd + 'render_options_vis.json')
        point = np.random.normal(0.0,20.0,(100,3))
        point_pcd = o3d.geometry.PointCloud()
        point_pcd.points = o3d.utility.Vector3dVector(point)
        self.pcd_source = copy.deepcopy(point_pcd)
        self.pcd_target = copy.deepcopy(point_pcd)
        self.vis_ICP.add_geometry(self.pcd_source)
        self.vis_ICP.add_geometry(self.pcd_target)
        # self.vis_ICP.update_renderer()


        # visualizer for SLAM
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.get_render_option().load_from_json(path_to_cwd + 'render_options_vis.json')
        load_view_point(path_to_cwd + 'viewpoint.json', self.vis) 
        add_geometry(self.vis, self.Loop_Closure.lineset)

        # make also a updating pyplot of the odometry and the keyframe positions

        # KALMAN FILTER for fusion with gps 
        # self.kalman_filter = KalmanFilter(dim_x=6, dim_z=6) # state is pos and vel, while measurement z is only pos
        # self.kalman_filter.x = np.array([[0.0,0.0,0.0,0.0,0.0,0.0]]).T
        # # state transition matrix
        # F = np.eye(6)

        # F[:3,3:] = np.eye(3)
        # self.kalman_filter.F = F
        # # H = np.zeros((3,6))
        # # H[:,3:] = np.eye(3)
        # self.kalman_filter.H = np.eye(6)
        # self.kalman_filter.P *= 10**2
        # self.kalman_filter.R = np.eye(6)*1.5
        # self.kalman_filter.Q = np.eye(6)*1.5
        # print(self.kalman_filter.Q)
        print('\n\n\n')



    def push_keyframe(self, init=False):
        self.keyframes.append(Keyframe(len(self.keyframes), self.pcd_kf, self.pcd, self.last_odometry, self.lidar_pose[:3,:3], self.cycle))
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(self.keyframes[-1].pose))
        self.keyframes[-1].update_slam_transform(self.pose_graph.nodes[-1].pose)
        self.keyframe_table.append([self.keyframes[-1].id, self.keyframes[-1].frame_idx])
        add_geometry(self.vis, self.keyframes[-1].slam_transformed_pcd) # keyframes point cloud
        add_geometry(self.vis, self.keyframes[-1].orientation_LineSet) # orientation rgb axes of keyframes

        if init:
            return
        # add egde to the graph, which is a connection between two nodes
        edge = o3d.pipelines.registration.PoseGraphEdge(
                                    self.keyframes[-1].frame_idx,
                                    self.keyframes[-2].frame_idx,
                                    self.last_transform,
                                    self.last_reg_information,
                                    uncertain = False)
        self.pose_graph.edges.append(edge)

        edge = o3d.pipelines.registration.PoseGraphEdge(
                                    self.cycle,
                                    (self.cycle - 1),
                                    self.last_odometry_transform,
                                    # self.last_reg_information,
                                    uncertain = False)
        self.pose_graph.edges.append(edge) # what? skal den ikke være her?

    def push_frame(self):
        # self.keyframes.append(Keyframe(len(self.keyframes), self.pcd_kf, self.pcd, self.last_odometry,self.lidar_pose[:3,:3], self.cycle))
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(self.last_odometry))
        # self.keyframes[-1].update_slam_transform(self.pose_graph.nodes[-1].pose)
        # self.keyframe_table.append([self.keyframes[-1].id, self.keyframes[-1].frame_idx])
        # add_geometry(self.vis, self.keyframes[-1].slam_transformed_pcd) # keyframes point cloud
        # add_geometry(self.vis, self.keyframes[-1].orientation_LineSet) # orientation rgb axes of keyframes

        # if init:
        #     return
        # add egde to the graph, which is a connection between two nodes
        edge = o3d.pipelines.registration.PoseGraphEdge(
                                    self.cycle,
                                    (self.cycle - 1),
                                    self.last_odometry_transform,
                                    # self.last_reg_information,
                                    uncertain = False)
        self.pose_graph.edges.append(edge)

    

    def update(self, pcd, pcd_ds, timestamp=None, gps_position=None, use_downsampled=True, downsample_keyframes=False):
        t1_process = time.perf_counter()
        self.timestamps.append(timestamp)
        self.update_delta_time()
        # self.gps_position = np.atleast_2d(gps_position).T

        # undistort according to last transfrom
        if self.undistort_pcd: pcd = undistort_pcd(pcd, self.last_odometry_transform, pitch_correction_deg=-self.euler_ang_lidar[0])
        if self.undistort_pcd: pcd_ds = undistort_pcd(pcd_ds, self.last_odometry_transform, pitch_correction_deg=-self.euler_ang_lidar[0])

        # transform according to lidar pose on the car
        pcd.transform(self.lidar_pose)
        pcd_ds.transform(self.lidar_pose)

        pcd = crop_pcd(pcd, z=(-2.5,100), r=(2, 500))
        pcd_ds = crop_pcd(pcd_ds, z=(-2.5,100), r=(2, 500))


        self.pcd = pcd
        self.pcd_kf = pcd_ds if downsample_keyframes  else pcd # keyframes are never downsampled if the others are not
        self.pcd_ds = pcd_ds if use_downsampled or downsample_keyframes else pcd
        # t1 = time.perf_counter()
        # self.pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size) if downsample else pcd # downsample the pcd
        # t2 = time.perf_counter()
        # print(f"downsample time: {t2 - t1:2f}s")
        
        #Current pcd vizualizer is updated 
        self.pcd_vis.points = self.pcd.points
        self.pcd_vis.normals = self.pcd.normals
        self.vis_pcd.update_geometry(self.pcd_vis)
        
        # if it is the first pcd, then create the first keyframe and return
        if len(self.keyframes) == 0:
            
            self.push_keyframe(init=True)
            self.Odometry.push_odometry(self.last_odometry, np.eye(4), init=True)

            # graphics added to open3d
            add_geometry(self.vis, self.Odometry.LineSet) # white odometry line
            add_geometry(self.vis, self.Loop_Closure.lineset) # purple lines between loop closure keyframes
            
            self.vis.poll_events()
            self.vis.update_renderer()
            return

        self.update_delta_time()
        

        # if a new keyframe was added it continues from here to loop closure and optimizing
        if self.update_keyframe():
            if self.use_loop_closure:
                self.detect_loop_proximity(dist=50)
                # self.repeated_proximity_detection()
                if self.confirm_loop_closure_candidates(idx = -1, manual=True): # meters distance for loop closure detection
                    if len(self.keyframes) - self.last_vis_update > 40:
                        self.optimize() # run optimize at loop closure
                        self.update_visual() #else update the last n
                        self.last_vis_update = len(self.keyframes)

        

        # self.vis.poll_events()
        # self.vis.update_renderer()
        # self.vis_loop.poll_events()
        # self.vis_loop.update_renderer()
        # self.vis_pcd.poll_events()
        # self.vis_pcd.update_renderer()
        # self.vis_ICP.poll_events()
        # self.vis_ICP.update_renderer()

        # laves til en seperat funktion der loop over vizulizer i en list? stor implementation!
        finish = False
        if not self.vis.poll_events():
            finish = True
        self.vis.update_renderer()
        if not self.vis_loop.poll_events():
            finish = True
        self.vis_loop.update_renderer()
        if not self.vis_pcd.poll_events():
            finish = True
        self.vis_pcd.update_renderer()
        if not self.vis_ICP.poll_events():
            finish = True
        self.vis_ICP.update_renderer()
        
        self.finish = finish

        # lock process at a frame x
        # x = 10
        # while self.cycle <= x:
        #     if not self.vis.poll_events():
        #         break
        #     self.vis.update_renderer()
        #     if not self.vis_loop.poll_events():
        #         break
        #     self.vis_loop.update_renderer()
        #     if not self.vis_pcd.poll_events():
        #         break
        #     self.vis_pcd.update_renderer()
        #     if not self.vis_ICP.poll_events():
        #         break
        #     self.vis_ICP.update_renderer()

        t2_process = time.perf_counter()
        process_time = t2_process - t1_process
        # print(f"process time:\t {process_time:2f}s")
        self.process_times.append(process_time)

        self.print_iteration_info()

        return self.finish

    

        
           
    # determines if keyframe should be added and add the keyframe
    def update_keyframe(self):

        keyframe_due = False
        # if self.cycle - self.keyframes[-1].frame_idx >= 10 and self.cycle !=0: # if more than 10 frames since the last keyframe, so if cycle - last keyframe id > 0
        #     keyframe_due = True # set to true if you want to force keyframe every n frame

        self.cycle += 1
        source = copy.deepcopy(self.pcd_ds)
        if self.use_local_map_icp:
            target  = self.generate_local_map(self.keyframes[-1].id) # get the previous local map instead of just the last pcd
            target.transform(np.linalg.inv(self.keyframes[-1].pose)) # center the pcd around the last position
        else:
            target = copy.deepcopy(self.keyframes[-1].pcd)
        # source = self.pcd_ds
        # target = self.keyframes[-1].pcd

        # draw_registration_result(source, traget , self.last_transform)
        t1_icp = time.perf_counter()
        reg, reg_information, evaluation = estimate_p2pl(source, target, init_trans=self.last_transform_guess, return_info=True, use_coarse_estimate=True, threshold=self.ICP_treshold)
        t2_icp = time.perf_counter()
        icp_time = t2_icp - t1_icp
        # print(f"ICP time:\t {icp_time:2f}s")
        self.icp_times.append(icp_time)
        transformation = copy.deepcopy(reg.transformation) # @ self.pitch_rot_bias
        # transformation[3,:3] = self.translation_bias

        self.pcd_source.points = source.transform(transformation).points
        self.pcd_target.points = target.points
        # self.vis_loop.visible = True
        self.vis_ICP.update_geometry(self.pcd_source.paint_uniform_color([1.0,0.0,0.0]))
        self.vis_ICP.update_geometry(self.pcd_target.paint_uniform_color([0.0,0.0,1.0]))
        


        self.fitnesses.append(reg.fitness)
        self.inlier_rmses.append(evaluation.inlier_rmse)
        # transformation_lidar = np.linalg.inv(self.lidar_pose) @ transformation @ (self.lidar_pose)
        # print(f"world transformation \n{transformation}")
        # print(f"Lidar transformation \n{transformation_lidar}")
        # print((transformation - transformation_lidar))
        # print(f"Lidar transformation \n{transformation}")

        # transformation, reg_information, evaluation = ground_plane_constrained_ICP(source, target, init_trans=self.last_transform_guess, return_info=True, threshold=self.ICP_treshold)

        # evaluation = o3d.pipelines.registration.evaluate_registration(source, target, self.ICP_treshold, reg.transformation )
        # transformation =  (self.lidar_pose) @ transformation @ np.linalg.inv(self.lidar_pose) # not
        # transformation =  np.linalg.inv(self.lidar_pose) @ transformation @ (self.lidar_pose)  # correct?
        # self.last_transformation = transformation

        self.last_reg_information = reg_information
        # draw_registration_result(source, target, transformation)

        # print(evaluation)

        # estimate angle and translation of transformation
        angle_of_transform = abs(angular_rotation_of_transform(transformation, is_deg=True, axis='z'))
        translation_of_transform = translation_dist_of_transform(transformation) 

        is_valid = (angle_of_transform) < self.keyframe_angle_threshold and translation_of_transform < self.keyframe_translation_threshold
        
        
        # calculate non-optimised odometry
        self.odometry_not_opt = self.keyframes[-1].odometry @ transformation

        # calculate optimised odometry
        # transformation =  np.linalg.inv(self.lidar_pose) @ transformation @ np.linalg.inv(self.z_rot_bias) @ (self.lidar_pose) 
        odometry =  self.keyframes[-1].pose @ transformation  # no correction here
        self.last_odometry_transform =  np.linalg.inv(self.Odometry.odometry[-1]) @ odometry 

        # check if the transformation is creating accelerations that are not possible and will use the last guess in that case
        # if not self.inertial_check():
        #     transformation = self.last_transform_guess
        #     odometry = self.keyframes[-1].pose @ transformation
    
        self.last_transform = transformation  # remember transform to use as guess, adding the last_odometry transform improves process time, also helps in case of bad registrations
        self.last_transform_guess = transformation @ self.last_odometry_transform # the initial guess

        self.last_odometry = odometry


        # save the odometry poses regardless, and update the 
        self.Odometry.push_odometry(self.last_odometry, self.odometry_not_opt) # use self.last_odometry
        self.vis.update_geometry(self.Odometry.LineSet)

        # SOME CODE TO MAKE THE CAMERA FOLLOW THE ODOMETRY

        # check if new transformation is within some limits of the previous keyframe else update the keyframe

        if self.cam_follow: transform_view_point(self.vis, self.last_odometry)

        is_good_registration = evaluation.inlier_rmse < self.ICP_treshold #and evaluation.fitness > 0.5
        if is_valid and not keyframe_due and is_good_registration: #reg.fitness > 0.7:
            self.push_frame()
            # self.prev_translation = translation_of_transform
            return False # False because the keyframe was not updated!

        # "else" make a new keyframe and add it to the pose graph using the transformation
        self.push_keyframe()
        
        # reset the last transformation
        self.last_transform_guess = self.last_odometry_transform # should improve itertation times by using the last movement as the next guess
        self.last_transform = np.eye(4)

        # print("keyframe added")
        return True  # as the keyframe was updated.


    def repeated_proximity_detection(self, look_back=7):

        self.recent_proximity_ids.append(set(self.proximity_idx))
        
        # print(self.recent_proximity_ids[-look_back:])
        if len(self.recent_proximity_ids[-look_back:]) > 0:
            recents = []
            for s in self.recent_proximity_ids[-look_back:]:
                recents = recents + list(s)
            repeated_prox = Counter(recents).most_common()
            # print(repeated_prox)
            if len(repeated_prox) == 0:
                return False
            self.proximity_idx = []
            for id, count in repeated_prox:
                if count >= (look_back - 1):
                    self.proximity_idx.append(id)

            self.look_back_idx = self.keyframes[-1].id - look_back// 2  # get the center id of the look_back range
            self.closest_idx = self.recent_proximity_ids[self.look_back_idx].intersection(set(self.proximity_idx))
            # print(self.closest_idx)
            # print(self.proximity_idx)
            if len(self.closest_idx) > 0: #
                self.proximity_idx = np.asarray(list(self.closest_idx))
                return True
        return False
        
        # closest_mask = distances[-1,proximity_idx] == np.min(distances[-1,proximity_idx]) # added, only look at the closest keyframe




    def detect_loop_proximity(self, dist):
       
        # distances = calculate_internal_distances(np.asarray(self.Odometry.positions), xy_only=True) # could be vectorized
        positions = [keyframe.position for keyframe in self.keyframes]
        distances = calculate_internal_distances(np.asarray(positions), xy_only=True) # could be vectorized
        proximity_idx = np.where(distances[-1,:] < dist**2)[0] # index of keyframes within dist meters i.e. r**2
        # proximity_idx_mask = proximity_idx < (self.cycle - 100) # the 100 latest frames are blocked from loop completion
        proximity_idx_mask = proximity_idx < (len(self.keyframes) - 25) # the 25 latest keyframesframes are blocked from loop completion
        # closest_mask = distances[-1, proximity_idx_mask ] == np.min(distances[-1, proximity_idx[proximity_idx_mask ]])
        self.proximity_idx = proximity_idx[proximity_idx_mask ]
        # print(f"loop closure proximities detected: {len(self.proximity_idx)}")
        self.proximites_count = len(self.proximity_idx)

        # if not any(proximity_idx_mask): # there are no proximities to earlier positions
            # t2_loop = time.perf_counter()    
            # print(f"loop detection time:\t{t2_loop - t1_loop}")
        return
        # closest_mask = distances[-1,proximity_idx] == np.min(distances[-1,proximity_idx]) # added, only look at the closest keyframe
        

    def confirm_loop_closure_candidates(self, idx=None, manual=False):
        if len(self.proximity_idx) == 0:
            return False

        accept_closure = False
        def kb_callback(inp):
            if inp.lower() == 'y':
                return True
            return False

        # NEED SOMETHING THAT CONFIRMS THE LOOP CLOSURE OVER THE NEEXT FRAMES
        # t1_loop = time.perf_counter()
        
        # proximity detection
        closure_added = False

        # keyframe_table = np.asarray(self.keyframe_table)
        fitnesses = []
        registrations = []
        inlier_rmses = []
        
        if idx is None:
            current_keyframe = self.keyframes[self.look_back_idx] # center of look back
        else:
            current_keyframe = self.keyframes[idx] # latest keyframe
        current_odometry = current_keyframe.pose
        # current_pcd = crop_pcd(copy.deepcopy(current_keyframe.pcd), z=z_crop) # latest keyframe's pcd, downsampled .transform(self.lidar_pose)
        current_pcd = copy.deepcopy(current_keyframe.pcd) # latest keyframe's pcd, downsampled .transform(self.lidar_pose)
        # for prox_idx in proximity_idx[closest_mask]:
        for prox_idx in self.proximity_idx:
            # find index of loop closure target and determine transformation, and add edge with uncertain=True
            # prox_keyframe_idx = keyframe_table[keyframe_table[:,1] == prox_idx, 0][0] # find a keyframe in the table that mathces the proximate frame
            prox_keyframe_idx = prox_idx
            # if len(prox_keyframe_idx) == 0:
            #     continue
            # current_pcd = copy.deepcopy(current_keyframe.pcd) # latest keyframe's pcd, downsampled
            closure_keyframe = self.keyframes[prox_keyframe_idx] # keyframe of the proximity
            # closure_pcd = crop_pcd(copy.deepcopy(closure_keyframe.pcd), z=z_crop )# pcd of the proximity .transform(self.lidar_pose)
            # closure_pcd = copy.deepcopy(closure_keyframe.pcd) # pcd of the proximity
            closure_pcd = self.generate_local_map(closure_keyframe.id) # local pcd of the proximity

            # Can only be used with strong odometry!!
            # get expected tranform between the 2 keyframes as the transformation between the odometry
            closure_odometry = closure_keyframe.pose
            # translation = closure_odometry[3,:3]
            closure_pcd.transform(np.linalg.inv(closure_odometry))
            init_trans = np.linalg.inv(closure_odometry) @ current_odometry
            init_trans[2,3] = 0.0  # ignore distance in z direction
            # init_trans[:3,3] = 0.0  # ignore distance

            # init_trans = np.eye(4)
            # print(init_trans)
            # print(current_odometry)
            # print(closure_odometry)

            reg_prox, reg_information, evaluation = estimate_p2pl(current_pcd, closure_pcd, init_trans=init_trans, return_info=True, use_coarse_estimate=True)
            # reg_prox, reg_information = estimate_p2pl_spin_init(closure_pcd, current_pcd, return_info=True, max_iteration=100)
            transformation_prox = reg_prox.transformation 
            closure_distance = translation_dist_of_transform(transformation_prox)
            # transformation_prox = np.linalg.inv(self.lidar_pose) @ reg_prox.transformation @ self.lidar_pose
            # print(transformation_prox)

            # evaluation = o3d.pipelines.registration.evaluate_registration(current_pcd, closure_pcd, self.ICP_treshold, transformation_prox)
            # draw_registration_result( current_pcd, closure_pcd, init_trans)
            # draw_registration_result( current_pcd, closure_pcd, transformation_prox)
            fitnesses.append(reg_prox.fitness)
            inlier_rmses.append(evaluation.inlier_rmse)
            registrations.append({"id":closure_keyframe.id,
                                  "frame_idx":closure_keyframe.frame_idx,
                                  "fitness":reg_prox.fitness, 
                                  'closure_distance': closure_distance,
                                  "inlier_rmse": evaluation.inlier_rmse, 
                                  "transformation":transformation_prox, 
                                  "information":reg_information,
                                  "pcd": closure_pcd,
                                  'keyframe': closure_keyframe})


        fitnesses = np.asarray(fitnesses)
        inlier_rmses = np.asarray(inlier_rmses)
        # print(fitnesses , inlier_rmses, sep="\n")

        # if np.argmin(inlier_rmses) != np.argmax(fitnesses):
        #     t2_loop = time.perf_counter()    
        #     print(f"loop detection time:\t{t2_loop - t1_loop}")
        #     return

        # best_registration = registrations[np.argmin(inlier_rmses)]
        best_registration = registrations[np.argmax(fitnesses)]


        # if best_registration['fitness'] > 0.5: 
        print(f"\x1B[2A\rBest inlier_rmse:\t{best_registration['inlier_rmse']:.4f}")
        print(f"Best fitness:\t{best_registration['fitness']:.4f}")

        is_valid = best_registration['inlier_rmse'] < 1 * self.ICP_treshold and best_registration['fitness'] > 0.5 and best_registration['closure_distance'] <= 10.0

        self.pcd1.points = current_pcd.transform(best_registration['transformation']).points
        self.pcd2.points = best_registration['pcd'].points
        # self.vis_loop.visible = True
        self.vis_loop.update_geometry(self.pcd1.paint_uniform_color([1.0,0.0,0.0]))
        self.vis_loop.update_geometry(self.pcd2.paint_uniform_color([0.0,0.0,1.0]))
        self.vis_loop.poll_events()
        self.vis_loop.update_renderer()
        if is_valid:
            keep_closure = False
            if best_registration['fitness'] > 0.4: # automatically added
                keep_closure = True
            # elif manual: # manual inspection
            #     kthread = KeyboardThread(kb_callback, mesg='Accept loop closure? "y" to accept: ')
            #     while True:
            #         # vis.update_geometry(src_pcd)
            #         if not self.vis_loop.poll_events():
            #             # vis.destroy_window()
            #             break
            #         self.vis_loop.update_renderer()
            #         if not self.vis.poll_events():
            #             # vis.destroy_window()
            #             break
            #         self.vis.update_renderer()
            #         if not ((keep_closure := kthread.retrieve()) is None): break
            #         # if not keep_closure := False: # bypass the manual decision

            if keep_closure:
                closure_added = True
                # print(f"loop closure added with transformation:\n{best_registration['transformation']}")
                self.pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(  current_keyframe.frame_idx,
                                                                                        best_registration["frame_idx"],
                                                                                        best_registration["transformation"],
                                                                                        best_registration["information"],
                                                                                        # confidence=reg_prox.fitness,
                                                                                        confidence=1.0,
                                                                                        uncertain=True))

                self.Loop_Closure.push_loop_closure(current_keyframe, best_registration["keyframe"])
                self.vis.update_geometry(self.Loop_Closure.lineset)
                # potential_closure_idx.add(prox_idx)
            # vis.destroy_window()
        # t2_loop = time.perf_counter()    
        # print(f"loop detection time:\t{t2_loop - t1_loop}")
        return closure_added

        #  plot the resgistration to manully validate

    


    def optimize(self):
        print("optimizing...")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                o3d.pipelines.registration.global_optimization(
                    self.pose_graph,
                    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                    self.optimization_options)


    def finalize(self):
        self.cycle += 1
        self.Odometry.push_odometry(self.last_odometry, self.odometry_not_opt)
        self.push_keyframe()
        if self.use_loop_closure:
            self.detect_loop_proximity(dist=10)
            # self.repeated_proximity_detection()
            self.confirm_loop_closure_candidates(idx=-1)
        self.optimize()
        self.update_visual(full=True)
        print('\x1B[2B')

    def update_visual(self, full=False):
        print("Updating visuals...")
        n_keyframes = len(self.keyframes)
        for i, (keyframe) in enumerate(self.keyframes):
            progress_bar.progress_bar(i, n_keyframes)

            keyframe.update_slam_transform(self.pose_graph.nodes[keyframe.frame_idx].pose, full=full)

            self.vis.update_geometry(keyframe.slam_transformed_pcd)
            self.vis.update_geometry(keyframe.orientation_LineSet)
            
            self.vis.poll_events()
            self.vis.update_renderer()

        self.Loop_Closure.update_lineset(self.keyframes)
        self.vis.update_geometry(self.Loop_Closure.lineset)
        self.Odometry.update_odometry(self.pose_graph.nodes)

        # self.Odometry.update_lineset()
        self.vis.update_geometry(self.Odometry.LineSet)

        self.vis.poll_events()
        self.vis.update_renderer()

        return





    def update_keyframe_nodes(self):
        for keyframe in self.keyframes:
            keyframe.pose_graph_node.pose = self.pose_graph.nodes[keyframe.id].pose
            # keyframe.odometry = np.linalg.inv(self.pose_graph.nodes[keyframe.id].pose)


    def generate_local_map(self, id, _range=1):
        # generates a local map around keyframe with chosen id, with range of nearby keyframes
        # the center keyframe should be located in origo?
        start = id - _range
        end = id + _range +1
        if start < 0:
            end += abs(start)
            start = 0
            if end > len(self.keyframes):
                end = len(self.keyframes)
        elif end > len(self.keyframes):
            end = len(self.keyframes)
            start = end - (2*_range + 1)
            if start < 0:
                start = 0
        local_map_pcd = o3d.geometry.PointCloud()
        try:
            for keyframe in self.keyframes[start: end]:
                local_map_pcd += keyframe.slam_transformed_pcd
        except: raise Exception("The range of keyframes for the local map is not available")
        
        return local_map_pcd
            


    def generate_map(self, full=True):
        print("generating map...")
        pcd_combined = o3d.geometry.PointCloud()
        for i, keyframe in enumerate(self.keyframes):
            progress_bar.progress_bar(i, len(self.keyframes))
            # keyframe.update_slam_transform(self.pose_graph.nodes[keyframe.frame_idx].pose, full=full)
            pcd = keyframe.slam_transformed_pcd_full if full else keyframe.slam_transformed_pcd
            pcd_combined += keyframe.slam_transformed_pcd_full
        # pcd_combined = o3d.voxel_down_sample(pcd_combined, 0.2)
        return pcd_combined


    def print_iteration_info(self):
        self.Odometry.positions[-1][0]
        print_info = {
            'cycle': self.cycle,
            'keyframes': len(self.keyframes),
            ' ' *5+'velocity': translation_dist_of_transform(self.last_odometry_transform, timestep=self.delta_time),
            ' '*10+'x': self.Odometry.positions[-1][0], 
            ' '*10+'y': self.Odometry.positions[-1][1], 
            ' '*10+'z': self.Odometry.positions[-1][2], 
            'ICP time': self.icp_times[-1],   
            'process time': self.process_times[-1], 
            'loop closure prox.': self.proximites_count
        }
        # self.proximites_count = 0
        print_dataframe = pd.DataFrame(print_info, index=[0])
        print(print_dataframe, end='\x1B[1A\r')


    def plot_pose_graph(self, ax=None):
        pos = []
        for node in self.pose_graph.nodes:
            pos.append(node.pose[:3,3])
        pos = np.asarray(pos)
        if ax is None:
            fig, ax = plt.subplots() 
        ax.plot(pos[:,0], pos[:,1], label='Pose graph', marker='.')
        ax.set_aspect('equal', adjustable='box')
        return ax

    def get_odometry(self):
        pos = np.asarray(self.Odometry.positions)
        return pos
    
    def get_odometry_non_opt(self):
        pos = []
        for pose in self.Odometry.non_opt_odometry:
            pos.append(pose[:3,3])
        pos = np.asarray(pos)
        return pos
    
    def get_poses(self):
        poses = []
        for pose in self.Odometry.odometry:
            poses.append(pose.ravel()[:-4])
        poses = np.asarray(poses)
        return poses

    def get_keyframe_poses(self):
        poses = []
        for keyframe in self.keyframes:
            poses.append(keyframe.pose.ravel()[:-4])
        poses = np.asarray(poses)
        return poses

    def get_fitnesses(self):
        return np.asarray(self.fitnesses)

    def get_inlier_rmses(self):
        return np.asarray(self.inlier_rmses)

    def get_timestamps(self):
        return np.asarray(self.timestamps)

    def get_process_times(self):
        return np.asarray(self.process_times)

    def get_icp_times(self):
        return np.asarray(self.icp_times)

    def plot_odometry(self, ax=None):
        pos = self.get_odometry() #np.asarray(self.Odometry.positions)
        if ax is None:
            fig, ax = plt.subplots() 
        ax.plot(pos[:,0], pos[:,1], label='Odometry')
        ax.set_aspect('equal', adjustable='box')
        return ax



    def plot_non_opt_odometry(self, ax=None):
        pos = []
        for pose in self.Odometry.non_opt_odometry:
            pos.append(pose[:3,3])
        pos = np.asarray(pos)
        if ax is None:
            fig, ax = plt.subplots() 
        ax.plot(pos[:,0], pos[:,1], label='Non Optimized Odometry')
        ax.set_aspect('equal', adjustable='box')
        return ax



    def debug_plane(self):

        yarea = (0, 10.0)
        xarea = (0, 10.0)
        X = np.linspace(*xarea, 1000)
        Y = np.linspace(*yarea, 1000)

        XX, YY = np.meshgrid(X,Y)
        XX = XX.ravel()
        YY = YY.ravel()
        # ZZ = np.ones_like(XX)* -2.0
        ZZ = np.zeros_like(XX)
        plane_np = np.c_[XX, YY, ZZ]

        plane = o3d.geometry.PointCloud()
        plane.points = o3d.utility.Vector3dVector(plane_np)

        add_geometry(self.vis, plane)


# KALMAN Filter

    def get_measurement_kalman(self):
            self.update_velocity()
            # self.update_position()
            # if self.gps_position is None:
            #     return
            z = np.r_[self.gps_position, self.velocity]
            # z = np.diag(z)
            return z
        
    def update_position(self):
        pos = self.last_odometry[:3,3] # should be gps
        self.position = np.atleast_2d(pos)
        
    def update_velocity(self, delta_t=0.2):
        translation = self.last_odometry_transform[:3,3]
        vel = translation/delta_t

        self.velocity = np.atleast_2d(translation).T

    def update_delta_time(self):
        if self.timestamps[-1] is None:
            self.delta_time = None
            return
        elif len(self.timestamps) <= 1:
            self.delta_time = 0.0
            return
        self.delta_time = self.timestamps[-1] - self.timestamps[-2]

    def inertial_check(self, max_acceleration=15.0, max_anglular_acc_deg = 45):
        dt = self.delta_time if not (self.delta_time is None) else 0.1
        acc = translation_dist_of_transform(self.previous_odometry_transform, dt) - translation_dist_of_transform( self.last_odometry_transform, dt)
        # acc = translation_dist_of_transform(acc_transform, timestep=dt)
        print(acc)
        # omega = angular_rotation_of_transform(acc_transform, timestep=dt, is_deg=True)
        if abs(acc) > max_acceleration:
            print("Extreme acceleration detected, using previous estimate!")
            self.last_odometry_transform = self.previous_odometry_transform
            # self.last_odometry_transform = np.linalg.inv(self.Odometry.odometry[-2]) @ self.Odometry.odometry[-2] 
            return False
        
        self.previous_odometry_transform = self.last_odometry_transform
        return True
            

if __name__ == "__main__":
    SLAM = PG_SLAM()


    # for i in range(500,N):
    for i in range(750, 1000):
        print(i)
        pcd = load_pcd_from_index(i, path_to_pcds, is_downsampled=False, voxel_size=1)
        SLAM.update(pcd,pcd, downsample_keyframes=True)

    print("done")
    SLAM.vis.run()