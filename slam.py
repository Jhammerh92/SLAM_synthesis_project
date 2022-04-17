import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys
import pyquaternion

from functions import *
from load_paths import * # loads paths to data


class Keyframe(object):
    def __init__(self, id, pcd, odometry, frame_idx, pose_graph):
        self.id = id                         # the corresponding frame index
        self.frame_idx = frame_idx
        self.odometry = copy.deepcopy(odometry)      # the odometry pose
        self.pcd = copy.deepcopy(pcd)        # point cloud data
        self.slam_transformed_pcd = copy.deepcopy(pcd)
        self.position = self.odometry[:3,3]


        self.pose_graph_node = o3d.pipelines.registration.PoseGraphNode(odometry)

        self.orientation_LineSet = o3d.geometry.LineSet()
        self.coord_points = []
        self.update_coordinate_frame()
        self.orientation_LineSet.points = o3d.utility.Vector3dVector(self.coord_points)
        self.orientation_LineSet.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
        self.orientation_LineSet.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        self.update_slam_transform(pose_graph)

    # updates the keyframes  after optimization
    def update_slam_transform(self, pose_graph):
        self.position =  self.pose_graph_node.pose[:3,3]
        self.odometry = self.pose_graph_node.pose
        # "reset" the transform
        self.slam_transformed_pcd.points = self.pcd.points
        self.slam_transformed_pcd.normals = self.pcd.normals
        # redo the transform after optimization
        self.slam_transformed_pcd.transform(self.pose_graph_node.pose)
        # self.slam_transformed_pcd.transform(pose_graph.nodes[self.id].pose)
        self.update_coordinate_frame()

    def update_coordinate_frame(self):
        position = self.pose_graph_node.pose[:3,3]
        self.coord_points.append(self.pose_graph_node.pose[:3,3])
        for row in self.pose_graph_node.pose[:3,:3].T:
            self.coord_points.append( row + position)
        
        

    



class Odometry(object):
    # use bundle adjustment from keyframe optimisation
    def __init__(self):
        self.odometry = []
        self.positions = []
        self.orientations = []
        self.lines = []
        self.odometry_LineSet = o3d.geometry.LineSet()

        self.coordinate_frame = [[0,0,0],[0, 0, 0], [1, 0, 0], [0, 0, 1]]
        self.coordinate_frame_lines = [[0, 1], [0, 2], [0, 3]]
        self.coordinate_frame_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        

    def update_odometry(self, pose, init=False):
        self.odometry.append(pose)
        self.positions.append(pose[:3,3])
        self.orientations.append(pose[:3,:3])
        if not init: self.update_lineset()

    def update_lineset(self):
        # pull the the updated positions
        self.odometry_LineSet.points = o3d.utility.Vector3dVector(self.positions)
        # self.lines.append([len(self.odometry_LineSet.points)-1 , len(self.odometry_LineSet.points)])
        self.lines = [[i,i+1] for i in range( len(self.odometry_LineSet.points)-1) ]
        # print(self.lines)
        self.odometry_LineSet.lines = o3d.utility.Vector2iVector(self.lines)
    
    # def new_coordinate_frame(self,odometry):


class Loop_Closure(object):
    def __init__(self):
        self.loop_closure_lineset = o3d.geometry.LineSet()
        self.lines = []
        self.positions = []

        self.loop_closure_lineset.lines = o3d.utility.Vector2iVector( self.lines)

        self.loop_closure_lineset.paint_uniform_color([1.0,0.0,1.0])
        


    def update_loop_closure_index(self, positions):
        for p in positions:
            self.positions.append(p)
        idx = len(self.positions)-1
        self.lines.append([idx-1, idx])
        
        self.loop_closure_lineset.points = o3d.utility.Vector3dVector(self.positions)
        self.loop_closure_lineset.lines = o3d.utility.Vector2iVector(self.lines)

        self.loop_closure_lineset.paint_uniform_color([1.0,0.0,1.0])

    



class PG_SLAM(object):
    def __init__(self):
        self.pose_graph = o3d.pipelines.registration.PoseGraph() # it has the pose graph
        self.cycle = 0
        

        # class handling the odometry part, but needs bundle adjustment to work with the pose graph optimization
        self.Odometry = Odometry()

        self.Loop_Closure = Loop_Closure()
        # self.Odometry.update_odometry(np.eye(4))
       
        self.keyframes = [] # an empty list for keyframes
        self.keyframe_table = []
        self.last_transform = np.eye(4) # the initial pose/transformation
        self.prev_translation = translation_dist_of_transform(self.last_transform)
        self.prev_rotation = angular_rotation_of_transform(self.last_transform)


        self.keyframe_angle_threshold = 15.0 # in degres
        self.keyframe_translation_threshold = 5.0 # in meters

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.get_render_option().load_from_json(path_to_cwd + 'render_options_vis.json')
        load_view_point(path_to_cwd + 'viewpoint.json', self.vis)
        add_geometry(self.vis, self.Loop_Closure.loop_closure_lineset)
        # self.vis.get_view_control().translate(100,100)

        # make also a updating pyplot of the odometry and the keyframe positions

        
        # set_view_point(self.vis, t=np.array([10,10,10]))


        self.optimization_options = o3d.pipelines.registration.GlobalOptimizationOption(
                                        max_correspondence_distance=0.5,
                                        edge_prune_threshold=0.25,
                                        preference_loop_closure=100.0,
                                        reference_node=0)


    def update(self, pcd):
        # pcd = pcd.voxel_down_sample(voxel_size=0.2) # downsample the pcd

        # if it is the first pcd, then create the first keyframe and return
        if len(self.keyframes) == 0:
            self.keyframes.append( Keyframe(0, pcd, np.eye(4), self.cycle, self.pose_graph))
            self.Odometry.update_odometry(np.eye(4), init=True)

            # self.vis.add_geometry(self.keyframes[-1].slam_transformed_pcd)
            add_geometry(self.vis, self.keyframes[-1].slam_transformed_pcd)
            add_geometry(self.vis, self.Odometry.odometry_LineSet)
            add_geometry(self.vis, self.keyframes[-1].orientation_LineSet)
            add_geometry(self.vis, self.Loop_Closure.loop_closure_lineset)
            
            # replace by an internal node instead of one handled in the keyframe?
            self.pose_graph.nodes.append(self.keyframes[-1].pose_graph_node)

            self.keyframe_table.append([self.keyframes[-1].id, self.keyframes[-1].frame_idx])
            self.vis.poll_events()
            self.vis.update_renderer()
            

            print("First keyframe at 0 is added")
            return
        

        # if a new keyframe was added it continues from here to loop closure and optimizing
        if not self.update_keyframe(pcd):
            return

        # detect loop closure
        if not self.detect_loop_proximity(pcd, 2): # meters dinstance for loop closure detection
            return
        
        self.update_visual(last_n=20) #else update the last n
        self.optimize() # run optimize at loop closure



        # temp = self.pose_graph.nodes[-1].pose
        # print(temp)
        # print(self.pose_graph.nodes[-1].pose)
        # print(temp - self.pose_graph.nodes[-1].pose)

        # self.update_keyframe_nodes()
        # self.update_visual() # update visual on all
        


        # OPTIMIZE
        # opt_n = 1
        # if len(self.keyframes) % opt_n == 0:
            # print("optimizing...")
            # self.optimize()

            # with o3d.utility.VerbosityContextManager(
            #     o3d.utility.VerbosityLevel.Debug) as cm:
            #         o3d.pipelines.registration.global_optimization(
            #             self.pose_graph,
            #             o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            #             o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            #             self.optimization_options)

            # for keyframe in self.keyframes[: -opt_n]:
            #     keyframe.update_slam_transform()
            #     self.vis.update_geometry(keyframe.pcd)
            #     self.vis.poll_events()
            #     self.vis.update_renderer()


        # else:
        #     for keyframe in self.keyframes[:-2]:
        #         print(keyframe.pose_graph_node.pose)
        #         keyframe.update_slam_transform()
        #         print(keyframe.pose_graph_node.pose)
        #         self.vis.update_geometry(keyframe.pcd)
        #         # load_view_point(path_to_cwd + 'viewpoint.json', self.vis)
        #         self.vis.poll_events()
        #         self.vis.update_renderer()
            

    # determines if keyframe should be added and add the keyframe
    def update_keyframe(self, pcd):
        keyframe_due = False
        if self.cycle - self.keyframes[-1].frame_idx >= 10 and self.cycle !=0: # if more than 10 frames since the last keyframe, so if cycle - last keyframe id > 0
            keyframe_due = True
        self.cycle += 1


        reg, reg_information = estimate_p2pl(self.keyframes[-1].pcd, pcd, self.last_transform, return_info=True)
        transformation = reg.transformation

        # estimate angle and translation of transformation
        angle_of_transform = angular_rotation_of_transform(transformation) 
        translation_of_transform = translation_dist_of_transform(transformation) 

        is_valid = abs(angle_of_transform) < self.keyframe_angle_threshold and translation_of_transform < self.keyframe_translation_threshold
        
        # NO CHECk WHETHER THE TRANSFORMATION IS GOOD!
        # if self.prev_translation != 0.0:
        #     deviation_from_prev = (translation_of_transform - self.prev_translation)
        #     is_valid = is_valid and abs(deviation_from_prev) < 1.0
        
        # calculate odometry
        odometry = self.keyframes[-1].odometry @ transformation
        # save the odometry poses regardless
        self.Odometry.update_odometry(odometry)
        self.vis.update_geometry(self.Odometry.odometry_LineSet)

        # SOME CODE TO MAKE THE CAMERA FOLLOW THE ODOMETRY
        transform_view_point(self.vis, odometry)

        # check if new transformation is within some limits of the previous keyframe else update the keyframe
        if is_valid and not keyframe_due:
            self.last_transform = transformation # accept the transformation
            self.prev_translation = translation_of_transform
            return False # False because the keyframe was not updated!

        # "else" make a new keyframe and add it to the pose graph using the transformation
        new_keyframe = Keyframe(len(self.keyframes), pcd, odometry, self.cycle, self.pose_graph)
        self.keyframes.append(new_keyframe)
        self.pose_graph.nodes.append(self.keyframes[-1].pose_graph_node)

        # update the visualizer
        #self.vis.add_geometry(self.keyframes[-1].slam_transformed_pcd)
        add_geometry(self.vis, self.keyframes[-1].slam_transformed_pcd) # this function saves the veiw point and reloads it after the geometry has been added
        add_geometry(self.vis, self.keyframes[-1].orientation_LineSet) # this function saves the veiw point and reloads it after the geometry has been added

        # add indexing to table so it can be found by the frame index
        self.keyframe_table.append([self.keyframes[-1].id, self.keyframes[-1].frame_idx])

        # reset the last transformation
        self.last_transform = np.eye(4) 

        # add egde to the graph, which is a connection between two nodes
        edge = o3d.pipelines.registration.PoseGraphEdge(
                                    self.keyframes[-1].id,
                                    self.keyframes[-2].id,
                                    transformation,
                                    reg_information,
                                    uncertain = False
        )
        self.pose_graph.edges.append(edge)


        print("keyframe added")
        return True  # as the keyframe was updated.


    



    def detect_loop_proximity(self, pcd, dist):
        # proximity detection
        # FIX to only check distance in x-y and not z?
        closure_added = False
        distances = calculate_internal_distances(np.asarray(self.Odometry.positions), xy_only=True)
        proximity_idx = np.where(distances[-1,:] < dist**2)[0] # within 10 meters i.e. r**2
        proximity_idx_mask = proximity_idx < (self.cycle - 100) # the 100 latest frames are blocked from loop completion
        proximity_idx = proximity_idx[proximity_idx_mask]

        if not any(proximity_idx_mask): # there are no proximities to earlier positions
            return
        
        keyframe_table = np.asarray(self.keyframe_table)
        print("loop closure proximity detected")
        for prox_idx in proximity_idx:
            # find index of loop closure target and determine transformation, and add edge with uncertain=True
            prox_keyframe_idx = keyframe_table[keyframe_table[:,1] == prox_idx, 0] # find a keyframe that mathces the proximate frame
            if len(prox_keyframe_idx) == 0:
                continue
            current_pcd = self.keyframes[-1].pcd # latest keyframe
            closure_pcd = self.keyframes[prox_keyframe_idx[0]].pcd # keyframe of the proximity

            # reg_prox, reg_information = estimate_p2pl(current_pcd, closure_pcd, return_info=True, use_coarse_estimate=True)
            reg_prox, reg_information = estimate_p2pl_spin_init(closure_pcd,current_pcd, return_info=True, max_iteration=100)
            transformation_prox = reg_prox.transformation
            

            # draw_registration_result( current_pcd, closure_pcd, transformation_prox)
            if reg_prox.fitness > 0.7: # reject loop closure if ICP fitness is less than 0.7
                closure_added = True
                print("loop closure added with fitness: {}".format(reg_prox.fitness))
                self.pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(  self.keyframes[-1].id,
                                                                                        prox_keyframe_idx,
                                                                                        transformation_prox,
                                                                                        reg_information,
                                                                                        confidence=reg_prox.fitness,
                                                                                        uncertain=True))

                self.Loop_Closure.update_loop_closure_index([self.keyframes[-1].position, self.keyframes[prox_keyframe_idx[0]].position])
                self.vis.update_geometry(self.Loop_Closure.loop_closure_lineset)
                # potential_closure_idx.add(prox_idx)
            
        return closure_added
        #  plot the resgistration to manully validate


    def optimize(self, optimize_all = False):
        print("optimizing...")
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
                o3d.pipelines.registration.global_optimization(
                    self.pose_graph,
                    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                    self.optimization_options)



    def update_visual(self, last_n=None):
        if last_n is None:
            for keyframe in self.keyframes:
                temp=keyframe.pose_graph_node.pose
                keyframe.update_slam_transform(self.pose_graph)
                temp2=keyframe.pose_graph_node.pose
                if not np.array_equal(temp, temp2):
                    print(keyframe.pose_graph_node.pose) # check if the loop closure updates the positions
                self.vis.update_geometry(keyframe.pcd)
                self.vis.poll_events()
                self.vis.update_renderer()
            return
        for keyframe in self.keyframes[-last_n:]:
                # keyframe.pose_graph_node.pose = self.pose_graph.nodes[keyframe.id].pose
                keyframe.update_slam_transform(self.pose_graph)
                self.vis.update_geometry(keyframe.pcd)
                self.vis.poll_events()
                self.vis.update_renderer()

    def update_keyframe_nodes(self):
        for keyframe in self.keyframes:
            keyframe.pose_graph_node.pose = self.pose_graph.nodes[keyframe.id].pose



    def generate_map():
        pcd_combined = o3d.geometry.PointCloud()
        for keyframe in self.keyframes:
            keyframe.update_slam_transform()
            pcd_combined += keyframe.slam_transformed_pcd()
        
        return o3d.voxel_down_sample(pcd_combined, 0.2)

            