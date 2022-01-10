# Import required packages
import cv2, time, uuid
from utils.app_utils import show_confirm_box
from utils.dir_handler import is_path_exists, get_path, make_dirs, get_dir_list


# Configure directories for image capture and storing train-test split images
def configure_directories(state):
    # Check and create path for collecting images if not exists
    images_path = get_path(state['APP_CONFIG']['image_capture']['dirs']['images'].split(' '))
    if not is_path_exists(images_path):
        make_dirs(images_path, False)
        folder = images_path.split("\\")[-1]
        print(f'* --> created {folder} directory')

    # Get image labels
    labels = state['APP_CONFIG']['image_capture']['labels']
    state['IMAGE_LABEL_PATHS'] = []
    for label in labels:
        label_path = get_path((images_path + ' ' + label).split(' '))
        if not is_path_exists(label_path):
            make_dirs(label_path)
            state['IMAGE_LABEL_PATHS'].append(label_path)
            print(f'* --> created {label} directory')

    # Check and create path for train images if not exists
    train_images_path = get_path(state['APP_CONFIG']['train_model']['dirs']['train_images'].split(' '))
    if not is_path_exists(train_images_path):
        make_dirs(train_images_path)
        folder = train_images_path.split("\\")[-1]
        print(f'* --> created {folder} directory')

    # Check and create path for test images if not exists
    test_images_path = get_path(state['APP_CONFIG']['train_model']['dirs']['test_images'].split(' '))
    if not is_path_exists(test_images_path):
        make_dirs(test_images_path)
        folder = test_images_path.split("\\")[-1]
        print(f'* --> created {folder} directory')

    return state


# Function to capture images
def capture_images(state):
    label_paths = state['IMAGE_LABEL_PATHS']
    if len(label_paths) >= 0:
        if len(label_paths) == 0:
            images_path = get_path(state['APP_CONFIG']['image_capture']['dirs']['images'].split(' '))
            label_paths = [get_path((images_path + ' ' + path).split(' ')) for path in get_dir_list(images_path)]
        for index in range(len(label_paths)):
            # Open camera for image capture
            cap = cv2.VideoCapture(0)
            label = label_paths[index].split('\\')[-1]
            print('* Collecting images for {}'.format(label))
            time.sleep(5)
            image_count = state['APP_CONFIG']['image_capture']['count_per_label']
            for image_num in range(image_count):
                print('* Collecting image {}'.format(image_num + 1))
                ret, frame = cap.read()
                image_path = get_path(
                    (label_paths[index] + ' ' + label + '.' + '{}.jpg'.format(str(uuid.uuid1()))).split(' '))
                cv2.imwrite(image_path, frame)
                cv2.imshow('frame', frame)
                time.sleep(2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    return state


# Function to capture training images
def capture_images_for_training(state):
    print('* Image Capture Module - Start')

    # Configure image capture directories
    # Configure directories
    print('*' + '-' * 89)
    print('* configuring paths and directories for image capture process')
    state = configure_directories(state)

    # Show alert dialog
    title = 'Capture Images'
    msg = 'Click Ok if you wish to capture images!!!'
    response = show_confirm_box(msg, title)
    if response == 'OK':
        print('*' + '-' * 89)
        print('* capturing images')
        # Capture images - system webcam
        state = capture_images(state)

    print('*')
    print('* Image Capture Module - End')
    print('*' * 90)
    return state
