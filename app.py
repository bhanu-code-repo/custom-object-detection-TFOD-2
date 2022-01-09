# Importing required packages
from core import capture_image, train_model
from utils.app_utils import ObjDict, configure_application
from core.capture_image import capture_images_for_training
from core.train_model import train_model_for_custom_object_detection
from core.object_detection import run_object_detection
from utils.app_enums import AppMode


# Defining main function
def main():
    # Define application state
    state = ObjDict()

    # Configure application
    state = configure_application(state)
    if state['IMAGE_CAPTURE_ENABLED']:
        state = capture_images_for_training(state)

    if state['MODEL_TRAINING_ENABLED']:
        state = train_model_for_custom_object_detection(state)

    if state['OBJECT_DETECTION_ENABLED']:
        state = run_object_detection(state)


# Run application
if __name__ == '__main__':
    try:
        main()
    except (ValueError, IndexError) as val_ind_error:
        print(f"There is a problem with values/parameters or dataset due to {val_ind_error}.")
