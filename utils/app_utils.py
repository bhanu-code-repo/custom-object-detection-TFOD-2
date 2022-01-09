# Importing required packages
import json
from utils.app_enums import get_enum
from utils.dir_handler import is_path_exists


# Object for dict
class ObjDict(dict):
    """
    Objdict class to conveniently store a state
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def load_application_configuration(state):
    if is_path_exists(state['CONFIG_FILE_NAME']):
        with open(state['CONFIG_FILE_NAME']) as config_file:
            try:
                state['APP_CONFIG'] = json.load(config_file)
                state['LOAD_CONFIG_STATUS'] = True
            except ValueError as error:
                print(error)

    return state


# Function to configure the application
def configure_application(state):
    # Define application configuration file name
    state['CONFIG_FILE_NAME'] = 'app_config.json'
    state['APP_CONFIG'] = {}
    state['LOAD_CONFIG_STATUS'] = False

    # Load application configuration
    state = load_application_configuration(state)

    # Update application configuration
    state['IMAGE_CAPTURE_ENABLED'] = False if state['APP_CONFIG'] == {} \
        else state['APP_CONFIG']['app_conf']['image_capture']
    state['MODEL_TRAINING_ENABLED'] = False if state['APP_CONFIG'] == {} \
        else state['APP_CONFIG']['app_conf']['train_model']
    state['OBJECT_DETECTION_ENABLED'] = False if state['APP_CONFIG'] == {} \
        else state['APP_CONFIG']['app_conf']['detect_object']

    # Define application operation state
    state['APP_MODE_ENABLED'] = get_enum('app_mode', 'debug') if state['APP_CONFIG'] == {} \
        else get_enum('app_mode', state['APP_CONFIG']['app_mode'])

    # Define object detection type
    state['OBJECT_DETECTION_TYPE'] = get_enum('object_detection_type', 'webcam') if state['APP_CONFIG'] == {} \
        else get_enum('object_detection_type', state['APP_CONFIG']['object_detection_type'])

    # Return application state information
    return state
