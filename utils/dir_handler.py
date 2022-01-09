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


def make_dirs(path, single_dir=True):
    os.mkdir(path) if single_dir else os.makedirs(path)


def get_dir_list(images_path):
    return os.listdir(images_path)


def get_files_list(path):
    return os.walk(path)


def copy_files(files, path, dest_path):
    for file in files:
        source = get_path([path, file])
        destination = get_path([dest_path, file])
        shutil.copy(source, destination)


def delete_files(data):
    for file in data[2]:
        file_path = os.path.join(data[0], file)
        os.remove(file_path)
