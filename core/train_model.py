# Importing required packages
from utils.app_utils import show_confirm_box
from utils.dir_handler import get_path, get_dir_list, get_files_list, copy_files, delete_files


# Function to perform train test split
def train_test_split(state):
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
            train_image_count = int((len(image_list) * 0.7) // 1)
            total_train_images += train_image_count
            train_image_list = image_list[0:train_image_count]

            # Copy train images and label xml file
            copy_files(train_image_list, path, train_img_dir_path)

            # Get test images and label xml file
            test_image_list = image_list[train_image_count:]
            total_test_images += len(test_image_list)

            # Copy test images and label xml file
            copy_files(test_image_list, path, test_img_dir_path)

    # Print train test split percentage
    total = total_train_images + total_test_images
    train_percentage = round((total_train_images/total)*100, 2)
    test_percentage = round((total_test_images/total)*100, 2)
    print(f'* train : {train_percentage}%, test : {test_percentage}%')

    return state


# Function to train pre-trained model(transfer learning) for detecting custom labels
def train_model_for_custom_object_detection(state):
    print('* Train Model Module - Start')

    # Display alert dialog
    title = 'Label Images'
    msg = 'Click Ok if you have completed image labeling!!!'
    response = show_confirm_box(msg, title)
    if response == 'OK':
        print('*' + '-' * 89)
        print('* performing train test split')
        # Perform train test split
        state = train_test_split(state)

    print('*')
    print('* Train Model Module - End')
    print('*' * 90)
    return state
