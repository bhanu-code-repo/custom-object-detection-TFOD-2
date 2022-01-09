# Importing required packages
import os


# Function to join paths
def join_path(*args):
    if len(args) == 2:
        return os.path.join(args[0], args[1])
    elif len(args) == 3:
        return os.path.join(args[0], args[1], args[2])
    elif len(args) == 4:
        return os.path.join(args[0], args[1], args[2], args[3])
    elif len(args) == 5:
        return os.path.join(args[0], args[1], args[2], args[3], args[4])
    elif len(args) == 6:
        return os.path.join(args[0], args[1], args[2], args[3], args[4], args[5])
    else:
        print('path args count out of range')


# Function to check if path exists
def is_path_exists(path):
    return True if os.path.exists(path) else False


