# Import required packages
import os, cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from utils.app_enums import ObjectDetectionType
from utils.app_enums import get_enum
from utils.dir_handler import get_path


# Function to configure paths and directories for object detection
def configure_path_directories(state):
    # Get model name
    model_name = state['APP_CONFIG']['train_model']['custom_model_name']

    # Get checkpoint path
    data = state['APP_CONFIG']['object_detection']['paths']['checkpoint']
    checkpoint = data.replace('<model_name>', model_name)
    state['checkpoint_path'] = get_path(checkpoint.split(' '))

    # Get pipeline config file path
    config_file = state['APP_CONFIG']['train_model']['files']['pipeline_config']
    data = state['APP_CONFIG']['object_detection']['paths']['config_file']
    config_file_path = data.replace('<model_name>', model_name).replace('<pipeline_config_file>', config_file)
    state['config_file_path'] = get_path(config_file_path.split(' '))

    # Get label file path
    label_file = state['APP_CONFIG']['train_model']['files']['label_map_file_name']
    data = state['APP_CONFIG']['object_detection']['paths']['label_file']
    label_file_path = data.replace('<label_file_name>', label_file)
    state['label_file_path'] = get_path(label_file_path.split(' '))

    return state


@tf.function
def detect_fn(image, state):
    """Detect objects in image."""

    image, shapes = state['detection_model'].preprocess(image)
    prediction_dict = state['detection_model'].predict(image, shapes)
    detections = state['detection_model'].postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


# Function to detect custom object
def run_object_detection(state):
    print('* Object Detection Module - Start')
    if get_enum('object_detection', state['APP_CONFIG']['object_detection']['type']) == ObjectDetectionType.WEBCAM:
        print('*'+'-' * 89)
        print('* object detection initiated using webcam ...')

        # Configure path/directories
        state = configure_path_directories(state)

        # Suppress TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')

        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(state['config_file_path'])
        model_config = configs['model']
        state['detection_model'] = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=state['detection_model'])
        ckpt.restore(state['checkpoint_path']).expect_partial()

        category_index = label_map_util.create_category_index_from_labelmap(state['label_file_path'],
                                                                            use_display_name=True)

        # Define the video stream
        cap = cv2.VideoCapture(0)

        while True:
            # Read frame from camera
            ret, image_np = cap.read()

            # Create tensor input
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections, predictions_dict, shapes = detect_fn(input_tensor, state)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)

            # Display output
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    print('*')
    print('* Object Detection Module - End')
    print('*' * 90)
    return state
