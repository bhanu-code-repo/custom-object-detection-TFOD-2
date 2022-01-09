# Import required packages
from utils.app_enums import ObjectDetectionType
from utils.app_enums import get_enum


# Function to detect custom object
def run_object_detection(state):
    print('* Object Detection Module - Start')
    if get_enum('object_detection_type', state['APP_CONFIG']['object_detection_type']) == ObjectDetectionType.WEBCAM:
        print('*'+'-' * 89)
        print('* object detection initiated ...')

    print('*')
    print('* Object Detection Module - End')
    print('*' * 90)
    return state
