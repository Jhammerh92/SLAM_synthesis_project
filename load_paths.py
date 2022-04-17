import os
import sys
from settings import *

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

# if __name__ == "__main__":
path_to_sequences = path_exists(get_paths_from_txt('path_to_sequences.txt')) + "/"
path_to_bins = path_to_sequences + SEQ + "/velodyne/"
path_to_pcds = path_to_bins + "pcds/"
path_to_cwd = os.getcwd() + "/"

if os.path.isdir(path_to_pcds):
    N = len(os.listdir(path_to_pcds))