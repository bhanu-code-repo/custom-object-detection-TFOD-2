# Importing required packages
import os, shutil


# Function to join paths
def get_path(path_list):
    if len(path_list) == 2:
        return os.path.join(path_list[0], path_list[1])
    elif len(path_list) == 3:
        return os.path.join(path_list[0], path_list[1], path_list[2])
    elif len(path_list) == 4:
        return os.path.join(path_list[0], path_list[1], path_list[2], path_list[3])
    elif len(path_list) == 5:
        return os.path.join(path_list[0], path_list[1], path_list[2], path_list[3], path_list[4])
    elif len(path_list) == 6:
        return os.path.join(path_list[0], path_list[1], path_list[2], path_list[3], path_list[4], path_list[5])
    else:
        print(f'path args count out of range: {path_list}')


# Function to check if path exists
def is_path_exists(path):
    return True if os.path.exists(path) else False


# Function to create directories
def make_dirs(path, single_dir=True):
    os.mkdir(path) if single_dir else os.makedirs(path)


# Function to get list of directories
def get_dir_list(images_path):
    return os.listdir(images_path)


# Function to get list of files
def get_files_list(path):
    return os.walk(path)


# Function to copy bulk images and label xml files
def copy_files_bulk(files, path, dest_path):
    for file in files:
        source = get_path([path, file])
        destination = get_path([dest_path, file])
        shutil.copy(source, destination)


# Function to copy images and label xml files
def copy_files(source, destination):
    shutil.copy(source, destination)


# Function to delete files from the directory
def delete_files(data, single_file=False):
    if not single_file:
        for file in data[2]:
            file_path = os.path.join(data[0], file)
            os.remove(file_path)
    else:
        os.remove(data)


# Function to execute command on command line terminal
def run_command(cmd):
    os.system(cmd)


def get_abs_path(path):
    return os.path.abspath(path)
