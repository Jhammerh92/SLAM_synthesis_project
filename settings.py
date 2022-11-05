#import os

DATA_SET = -1 # 0 for DTU, 1 for KITTI
SEQ = 0       # sequence to work on

DATA_SETS = ['DTU', 'KITTI', 'DTU']
DATA_SET_LABEL = DATA_SETS[DATA_SET]
SEQ_LABEL = "{:02}".format(SEQ) # sequence as zeropadded string
DATA_LABEL = f'{DATA_SET_LABEL}_{SEQ_LABEL}'

VOXEL_SIZE = 1.0

START_INDEX = 0 # 0 from start
END_INDEX = None # or None for end


UNDISTORT_PCD = DATA_SET == 0 or DATA_SET == -1
USE_LOOP_CLOSURE = True
USE_LOCAL_MAP_ICP = True

DOWNSAMPLE_KEYFRAMES = True
DOWNSAMPLE_REST = True

# OUTLIER_TYPE = 'STD' # 'RADIUS'



CAM_FOLLOW = True



