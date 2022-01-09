# Importing required packages
from utils.app_utils import ObjDict, configure_application
from core.capture_image import capture_images_for_training
from core.train_model import train_model_for_custom_object_detection
from core.object_detection import run_object_detection


# Defining main function
def main():
    # Welcome message
    print('*' * 90)
    print('* Welcome to custom object detection application using TFOD 2.0 API')
    print('*' * 90)

    # Define application state
    state = ObjDict()

    # Configure application
    state = configure_application(state)
    if state['LOAD_CONFIG_STATUS']:
        if state['APP_CONFIG']['app_conf']['image_capture']:
            state = capture_images_for_training(state)

        if state['APP_CONFIG']['app_conf']['train_model']:
            state = train_model_for_custom_object_detection(state)

        if state['APP_CONFIG']['app_conf']['detect_object']:
            state = run_object_detection(state)
    else:
        print('error loading application configuration file ...')


# Run application
if __name__ == '__main__':
    try:
        main()
    except (ValueError, IndexError) as val_ind_error:
        print(f"There is a problem with values/parameters or dataset due to {val_ind_error}.")
