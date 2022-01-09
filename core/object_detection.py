# Import required packages
from utils.app_enums import ObjectDetectionType


# Function to detect custom object
def run_object_detection(state):
    if state['OBJECT_DETECTION_TYPE'] == ObjectDetectionType.WEBCAM:
        print('object detection initiated ...')

    return state
