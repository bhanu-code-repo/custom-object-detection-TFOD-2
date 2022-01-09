# Import required packages
import enum


# Application operational mode enum definition
class AppMode(enum.Enum):
    DEBUG = 1
    PROD = 2


# Object detection type enum definition
class ObjectDetectionType(enum.Enum):
    WEBCAM = 1
    IMAGE = 2
    VIDEO = 3


# Function to map application configuration to enum values
def get_enum(key, value):
    if key.lower() == 'app_mode':
        if value.lower() == 'debug':
            return AppMode.DEBUG
        if value.lower() == 'prod':
            return AppMode.PROD

    if key.lower() == 'object_detection_type':
        if value.lower() == 'webcam':
            return ObjectDetectionType.WEBCAM
        if value.lower() == 'image':
            return ObjectDetectionType.IMAGE
        if value.lower() == 'video':
            return ObjectDetectionType.VIDEO