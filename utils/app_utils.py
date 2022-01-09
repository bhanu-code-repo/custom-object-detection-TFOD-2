# Importing required packages
import json, pyautogui
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


# Function to read application configuration file
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
    state['IMAGE_LABEL_PATHS'] = []

    # Load application configuration
    state = load_application_configuration(state)

    # Return application state information
    return state


# Function to show user alert dialog
def show_confirm_box(msg, title):
    return pyautogui.confirm(msg, title)
