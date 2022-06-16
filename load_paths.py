import os
import sys
from numpy import loadtxt, eye
from settings import *
import pandas as pd

# read path to data from the .txt file
def get_paths_from_txt(txt_with_paths):
    file_with_paths = open(txt_with_paths, 'r')
    lines = file_with_paths.read().splitlines()
    return lines

# return the first path that exists
def path_exists(paths):
    for path in paths:
        if os.path.isdir(path):
            return path
    raise Exception(f'None of {paths} exists')

def file_exists(files):
    for file in files:
        if os.path.isfile(file):
            return file
    raise Exception(f'None of {files} exists')

def listdirs(folder):
    dirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    # dirs.sort(key=lambda x: "{}".format(int(x.replace('_',''))))
    return dirs


# def set_paths_to_data_from(data_set, seq):
paths_to_datasets = get_paths_from_txt(os.getcwd() + r'/path_to_sequences.txt')

path_to_sequences = path_exists([paths_to_datasets[DATA_SET]])

try:
    sequence_folder = listdirs(path_to_sequences)
    sequence_folder.sort(key=lambda x: "{:06}".format(int(x.replace('_',''))))
    sequence_folder = sequence_folder[SEQ]
except: raise Exception(f"Sequence number is out range! {SEQ}")

SEQUENCE_NAME = sequence_folder
PATH_TO_SEQUENCE = f"{path_to_sequences}/{sequence_folder}"
path_to_bins = f"{path_to_sequences}/{sequence_folder}/bins/"
PATH_TO_PCDS = f"{path_to_sequences}/{sequence_folder}/pcds/"
try:
    path_to_pose_txt = file_exists([f"{PATH_TO_SEQUENCE}/pose.csv",
                                 f"{PATH_TO_SEQUENCE}/pose.txt"]) 
except:
    path_to_pose_txt = ''

path_to_cwd = os.getcwd() + "/"


try:
    os.mkdir(f'{PATH_TO_SEQUENCE}/processed_data')
    print("Processed data folder is created..")
except:
    print("Processed data already exists..")
PATH_TO_PROCESSED = f'{PATH_TO_SEQUENCE}/processed_data'
    

    # path_to_pose = path_exists(get_paths_from_txt(r'/Users/JHH/Library/Mobile Documents/com~apple~CloudDocs/DTU/10Semester/Syntese/SLAM_synthesis_project/path_to_poses.txt')) + "/" + SEQ + ".txt"


# set_paths_to_data_from(DATA_SET, SEQ)
try:
    CALIB_LIDAR_POSE = loadtxt(f'{path_to_sequences}/calibrated_lidar_pose.txt') # should be loaded with the data set...
    CALIB_BOOL = True
except:
    CALIB_LIDAR_POSE = eye(4) # if no calibrationa exists
    CALIB_BOOL = False

TOTAL_PCDS = len([file for file in os.listdir(PATH_TO_PCDS) if file.endswith('.pcd')])










end = TOTAL_PCDS if END_INDEX == None else END_INDEX

SETTINGS_DICT = {
    'label':                DATA_LABEL,
    'voxel size':           VOXEL_SIZE,
    'use loop closure':     USE_LOOP_CLOSURE,
    'local map ICP':        USE_LOCAL_MAP_ICP,
    'undistort_pcd':        UNDISTORT_PCD,
    'lidar calibration':    CALIB_BOOL,
    'index (start, end)':    f'{str(START_INDEX)}, {end}'
}

SETTINGS = pd.DataFrame(SETTINGS_DICT.values(),index=SETTINGS_DICT.keys(),columns=['value'])

slam_params = {
    'calib_lidar_pose': CALIB_LIDAR_POSE,
    'undistort_pcd': UNDISTORT_PCD,
    'cam_follow': CAM_FOLLOW,
    'use_loop_closure': USE_LOOP_CLOSURE,
    'use_local_map_icp': USE_LOCAL_MAP_ICP
}




# if os.path.isdir(path_to_pcds):
#     N = len(os.listdir(path_to_pcds))

