# Importing required packages
import requests, tarfile
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from utils.app_utils import show_confirm_box
from utils.dir_handler import (get_path, get_dir_list, get_files_list, copy_files_bulk, delete_files, is_path_exists,
                               make_dirs, get_abs_path, run_command, copy_files)


# Function to configure paths and directories for model training
def configure_paths_directories(state):
    print('*' + '-' * 89)
    print('* configuring paths and directories for model training process')

    # Get script directory path
    scripts_path = get_path((state['APP_CONFIG']['train_model']['dirs']['scripts']).split(' '))
    if not is_path_exists(scripts_path):
        make_dirs(scripts_path)
        folder = scripts_path.split("\\")[-1]
        print(f'* --> created {folder} directory')

    # Get annotation directory path
    annot_path = get_path((state['APP_CONFIG']['train_model']['dirs']['annotation']).split(' '))
    if not is_path_exists(annot_path):
        make_dirs(annot_path)
        folder = annot_path.split("\\")[-1]
        print(f'* --> created {folder} directory')

    # Get pre-trained model directory path
    pre_trained_model_path = get_path((state['APP_CONFIG']['train_model']['dirs']['pre-trained_model']).split(' '))
    if not is_path_exists(pre_trained_model_path):
        make_dirs(pre_trained_model_path, False)
        folder = pre_trained_model_path.split("\\")[-1]
        print(f'* --> created {folder} directory')

    # Create path for my model
    my_model = state['APP_CONFIG']['train_model']['custom_model_name']
    model_dir = get_path((state['APP_CONFIG']['train_model']['dirs']['model'] + ' ' + my_model).split(' '))
    if not is_path_exists(model_dir):
        make_dirs(model_dir)
        print(f'* --> created {my_model} directory')

    # For my model checkpoint
    checkpoint_path = state['APP_CONFIG']['train_model']['dirs']['model'] + ' ' + my_model + ' ' + 'ckpt-11'
    state['checkpoint_path'] = get_path(checkpoint_path.split(' '))

    # Create paths for files
    pipeline_config_file_name = state['APP_CONFIG']['train_model']['files']['pipeline_config']
    # For current model
    file_path = state['APP_CONFIG']['train_model']['dirs']['model'] + ' ' + my_model + ' ' + pipeline_config_file_name
    state['pipeline_config_file_path'] = get_path(file_path.split(' '))

    # For pre-trained model
    pre_trained_model_name = state['APP_CONFIG']['train_model']['pre-trained_model_name']
    file_path = state['APP_CONFIG']['train_model']['dirs'][
                    'pre-trained_model'] + ' ' + pre_trained_model_name + ' ' + pipeline_config_file_name
    state['pre-trained_model_pipeline_config_file_path'] = get_path(file_path.split(' '))

    # For tensorflow record generation script
    tf_record_file_name = state['APP_CONFIG']['train_model']['files']['tf_record_script_name']
    state['tf_record_script_file_path'] = get_path((scripts_path + ' ' + tf_record_file_name).split(' '))

    # For label map file
    label_map_file_name = state['APP_CONFIG']['train_model']['files']['label_map_file_name']
    state['label_map_file_path'] = get_path((annot_path + ' ' + label_map_file_name).split(' '))

    # Update process status
    state['PATH_DIRS_CONFIGURED'] = True

    # Initialize next process flags
    state['PRE_TRAINED_MODEL_AVAILABLE'] = False
    state['TF_GEN_SCRIPT_AVAILABLE'] = False
    state['LABEL_MAP_FILE_CREATED'] = False
    state['TRAIN_TEST_SPLIT_CREATED'] = False
    state['TRAIN_TEST_REC_FILE_CREATED'] = False
    state['CONFIG_FILE_UPDATED'] = True

    return state


# Function to download pre-trained model from url
def download_pre_trained_model(state):
    print('*' + '-' * 89)
    print('* checking for model ...')
    pre_trained_model_name = state['APP_CONFIG']['train_model']['pre-trained_model_name']
    model_path = get_path((state['APP_CONFIG']['train_model']['dirs']['pre-trained_model'] + ' ' +
                           pre_trained_model_name).split(' '))
    if not is_path_exists(model_path):
        print('* downloading pre-trained model ...')
        model_url = state['APP_CONFIG']['train_model']['pre-trained_model_url']
        # Make a GET request to download file
        response = requests.get(model_url, stream=True)

        # Extract downloaded file
        print('* extracting pre-trained model ...')
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        path = '\\'.join(model_path.split('\\')[:-1])
        file.extractall(path=path)
        print('* download and extraction completed.')
    else:
        print('* pre-trained model exists')

    # Update process status
    state['PRE_TRAINED_MODEL_AVAILABLE'] = True

    return state


# Function to download tensorflow record generation file from url
def download_tf_record_gen_script(state):
    print('*' + '-' * 89)
    print('* checking for script ...')
    script_path = state['tf_record_script_file_path']
    if not is_path_exists(script_path):
        script_name = script_path.split('\\')[-1]
        print(f'* downloading {script_name} file ...')

        # Make a GET request to download file
        script_url = state['APP_CONFIG']['train_model']['tf_record_script_url']
        response = requests.get(script_url, stream=True)
        with open(script_path, 'wb') as f:
            f.write(response.content)
        print('* download completed.')
    else:
        print('* script found ...')

    # Update process status
    state['TF_GEN_SCRIPT_AVAILABLE'] = True

    return state


# Function to create label map file
def create_label_map_file(state):
    print('*' + '-' * 89)
    print('* checking for label file ...')
    label_file = state['label_map_file_path']
    if is_path_exists(label_file):
        delete_files(label_file, True)
        print('* deleting existing label file ...')

    # Create file
    labels = state['APP_CONFIG']['train_model']['labels']
    label_id = 0
    label_list = []
    for label in labels:
        label_id += 1
        label_list.append({'name': label, 'id': label_id})
    with open(label_file, 'w') as f:
        for label in label_list:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

    print('* new label file created ...')

    # Update process status
    state['LABEL_MAP_FILE_CREATED'] = True

    return state


# Function to perform train test split
def train_test_split(state):
    print('*' + '-' * 89)
    print('* performing train test split ...')

    # Remove any exiting images from the train directory
    train_img_dir_path = get_path(state['APP_CONFIG']['train_model']['dirs']['train_images'].split(' '))
    train_data = [f for file in get_files_list(train_img_dir_path) for f in file]
    if len(train_data[2]) > 0:
        delete_files(train_data)

    # Remove any exiting images from the test directory
    test_img_dir_path = get_path(state['APP_CONFIG']['train_model']['dirs']['test_images'].split(' '))
    test_data = [f for file in get_files_list(test_img_dir_path) for f in file]
    if len(test_data[2]) > 0:
        delete_files(test_data)

    label_paths = state['IMAGE_LABEL_PATHS']
    total_train_images = 0
    total_test_images = 0
    if len(label_paths) >= 0:
        if len(label_paths) == 0:
            images_path = get_path(state['APP_CONFIG']['image_capture']['dirs']['images'].split(' '))
            label_paths = [get_path((images_path + ' ' + path).split(' ')) for path in get_dir_list(images_path)]
        for path in label_paths:
            image_list = [f for (root, dirs, file) in get_files_list(path) for f in file]

            # Get train images and label xml file
            count = int((len(image_list) * 0.7) // 1)
            train_image_count = count if count % 2 == 0 else count - 1
            total_train_images += train_image_count
            train_image_list = image_list[0:train_image_count]

            # Copy train images and label xml file
            copy_files_bulk(train_image_list, path, train_img_dir_path)

            # Get test images and label xml file
            test_image_list = image_list[train_image_count:]
            total_test_images += len(test_image_list)

            # Copy test images and label xml file
            copy_files_bulk(test_image_list, path, test_img_dir_path)

    # Print train test split percentage
    total = total_train_images + total_test_images
    train_percentage = round((total_train_images / total) * 100, 2)
    test_percentage = round((total_test_images / total) * 100, 2)
    print(f'* train : {train_percentage}%, test : {test_percentage}%')

    # Update process status
    state['TRAIN_TEST_SPLIT_CREATED'] = True

    return state


# Function to generate train test tensorflow record files
def generate_tensorflow_record_files(state):
    print('*' + '-' * 89)
    print('* generating tensorflow record files ...')
    train_rec_file_flag = False
    test_rec_file_flag = False

    # Get label file path
    label_file = get_abs_path(state['label_map_file_path'])
    script_path = get_abs_path(state['tf_record_script_file_path'])

    # Get train record file path
    train_file = state['APP_CONFIG']['train_model']['dirs']['annotation'] + ' ' + 'train.record'
    train_rec_file = get_abs_path(get_path(train_file.split(' ')))
    if is_path_exists(train_rec_file):
        delete_files(train_rec_file, True)
        print('* deleting existing train.record file')

    # Get train image files list
    train_images = get_path(state['APP_CONFIG']['train_model']['dirs']['train_images'].split(' '))
    train_data = [f for file in get_files_list(train_images) for f in file]
    if len(train_data[2]) > 0:
        print('* creating tf record file for train set')
        command = f'python {script_path} -x {get_abs_path(train_images)} -l {label_file} -o {train_rec_file}'
        run_command(command)
        train_rec_file_flag = True
    else:
        print('* no train images found for creating train.record file')

    # Get test record file path
    test_file = state['APP_CONFIG']['train_model']['dirs']['annotation'] + ' ' + 'test.record'
    test_rec_file = get_abs_path(get_path(test_file.split(' ')))
    if is_path_exists(test_rec_file):
        delete_files(test_rec_file, True)
        print('* deleting existing test.record file')

    test_images = get_path(state['APP_CONFIG']['train_model']['dirs']['test_images'].split(' '))
    test_data = [f for file in get_files_list(test_images) for f in file]
    if len(test_data[2]) > 0:
        print('* creating tf record file for test set')
        command = f'python {script_path} -x {get_abs_path(test_images)} -l {label_file} -o {test_rec_file}'
        run_command(command)
        test_rec_file_flag = True
    else:
        print('* no train images found for creating test.record file')

    # Update process status
    state['TRAIN_TEST_REC_FILE_CREATED'] = True if (train_rec_file_flag and test_rec_file_flag) else False

    return state


# Function to update exiting pipeline.config file
def update_model_config_for_transfer_learning(state):
    print('*' + '-' * 89)
    print('* copying and updating pipeline.config file ...')

    # Get file paths
    pre_trained_file_path = state['pre-trained_model_pipeline_config_file_path']
    current_file_path = state['pipeline_config_file_path']

    if is_path_exists(current_file_path):
        delete_files(current_file_path, True)
        print('* deleting existing pipeline.config file')

    # Copy files
    copy_files(pre_trained_file_path, current_file_path)

    # Read pipeline config file
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(current_file_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    # Get data
    pre_trained_model_name = state['APP_CONFIG']['train_model']['pre-trained_model_name']
    model_checkpoint_path = get_path((state['APP_CONFIG']['train_model']['dirs']['pre-trained_model'] + ' ' +
                                      pre_trained_model_name + ' ' + 'checkpoint' + ' ' + 'ckpt-0').split(' '))
    train_file = state['APP_CONFIG']['train_model']['dirs']['annotation'] + ' ' + 'train.record'
    train_rec_file_path = get_path(train_file.split(' '))
    test_file = state['APP_CONFIG']['train_model']['dirs']['annotation'] + ' ' + 'test.record'
    test_rec_file_path = get_path(test_file.split(' '))

    # Update config parameters
    labels = state['APP_CONFIG']['train_model']['labels']
    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = model_checkpoint_path
    pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
    pipeline_config.train_input_reader.label_map_path = state['label_map_file_path']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [train_rec_file_path]
    pipeline_config.eval_input_reader[0].label_map_path = state['label_map_file_path']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [test_rec_file_path]

    # Update config file
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(current_file_path, "wb") as f:
        f.write(config_text)

    print('* pipeline.config file updated!')

    # Update process status
    state['CONFIG_FILE_UPDATED'] = True

    return state


# Function to train pre-trained model for custom labels
def train_model(state):
    steps = ['PATH_DIRS_CONFIGURED', 'PRE_TRAINED_MODEL_AVAILABLE', 'TF_GEN_SCRIPT_AVAILABLE', 'LABEL_MAP_FILE_CREATED',
             'TRAIN_TEST_SPLIT_CREATED', 'TRAIN_TEST_REC_FILE_CREATED', 'CONFIG_FILE_UPDATED']
    map_flag_status = lambda flag: 1 if flag else 0
    flag_status = [map_flag_status(state[flag]) for flag in steps]
    if sum(flag_status) == len(steps):
        print('* training model ...')
        script_name = state['APP_CONFIG']['train_model']['files']['model_training_script']
        training_script_path = get_path(['object_detection', script_name])

        my_model = state['APP_CONFIG']['train_model']['custom_model_name']
        checkpoint_path = state['APP_CONFIG']['train_model']['dirs']['model'] + ' ' + my_model
        model_dir = get_path(checkpoint_path.split(' '))
        config_file_path = state['pipeline_config_file_path']

        # Create command
        command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=10000"\
            .format(training_script_path, model_dir, config_file_path)

        # Run model training command
        run_command(command)

    else:
        status_tup = [(flag,name) for flag, name in zip(flag_status, steps)]
        process_status = ''
        for status in status_tup:
            flag_val, step_name = status
            process_status += f'{step_name}: {flag_val}, '
        print('* error: complete steps with flag value 0')
        print(f'* {process_status[:-2]}')

    return state


# Function to train pre-trained model(transfer learning) for detecting custom labels
def train_model_for_custom_object_detection(state):
    print('* Train Model Module - Start')

    # Configure directories
    state = configure_paths_directories(state)

    # Download pre-trained model
    state = download_pre_trained_model(state)

    # Download tensorflow record generation script
    state = download_tf_record_gen_script(state)

    # Create label map file for training
    state = create_label_map_file(state)

    # Display alert dialog
    title = 'Labeling Custom Images'
    msg = 'Click Ok if you have completed image labeling!!!'
    response = show_confirm_box(msg, title)
    if response == 'OK':
        # Perform test train-test split
        state = train_test_split(state)

        # Generate tensorflow record file for train and test images
        state = generate_tensorflow_record_files(state)

        # Update pipeline config file for transfer learning
        state = update_model_config_for_transfer_learning(state)

        # Training model for custom images
        state = train_model(state)

    print('*')
    print('* Train Model Module - End')
    print('*' * 90)
    return state
