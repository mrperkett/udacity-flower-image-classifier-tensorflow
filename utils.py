def import_tensorflow():
    """
    Import tensorflow and disable all warnings.
    
    Copy and paste from
    https://weepingfish.github.io/2020/07/22/0722-suppress-tensorflow-warnings/
    """
    # Filter tensorflow version warnings
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf

tf = import_tensorflow()

import json
import logging
import numpy as np
import os
from PIL import Image

# define image augmentation and preprocessing layers outside of the function as
# is required by tensorflow variable Singelton pattern
# https://www.tensorflow.org/guide/function#creating_tfvariables
img_height = 224
img_width = 224
image_augmentation_layers = [
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomZoom(0.2, 0.2, fill_mode="nearest"),
        tf.keras.layers.RandomRotation([-1./12, 1./12])
        ]

image_preprocessing_layers = [
        tf.keras.layers.Resizing(height=256, width=256, crop_to_aspect_ratio=True),
        tf.keras.layers.CenterCrop(height=img_height, width=img_width),
        tf.keras.layers.Rescaling(1./255)
        ]

def get_class_idx_and_name_mappings(dataset_info):
    """
    Get the mapping between class_idx (e.g. 0) and class_name (e.g. "pink primrose")
    using the TensorFlow DataSetInfo.
    """
    # build the mapping dictionaries using the .features["label"].names list values for
    # class_name and indices for class_idx
    class_idx_to_class_name = dict()
    class_name_to_class_idx = dict()
    for class_idx, class_name in enumerate(dataset_info.features["label"].names):
        if class_idx in class_idx_to_class_name:
            raise ValueError(f"class_idx ({class_idx}) is repeated")
        if class_name in class_name_to_class_idx:
            raise ValueError(f"class_name ({class_name}) is repeated")
        
        class_idx_to_class_name[class_idx] = class_name
        class_name_to_class_idx[class_name] = class_idx

    # do a quick check to verify the mapping matches internal functions
    for class_idx, class_name in class_idx_to_class_name.items():
        if dataset_info.features["label"].int2str(class_idx) != class_name:
            raise AssertionError(f"class_idx ({class_idx}) does not map to class_name ({class_name}) as expected")
        if dataset_info.features["label"].str2int(class_name) != class_idx:
            raise AssertionError(f"class_name ({class_name}) does not map to class_idx ({class_idx}) as expected")

    return class_idx_to_class_name, class_name_to_class_idx


def get_class_idx_and_name_mappings_from_json(json_file_path):
    """
    Get the mapping between class_idx (e.g. 0) and class_name (e.g. "pink primrose")
    using a json input file.
    """
    class_idx_to_class_name = dict()
    class_name_to_class_idx = dict()
    with open(json_file_path, "r") as inp_file:
        for class_idx_str, class_name in json.load(inp_file).items():
            class_idx = int(class_idx_str)
            if class_idx in class_idx_to_class_name:
                raise ValueError(f"class_idx ({class_idx}) is repeated")
            if class_name in class_name_to_class_idx:
                raise ValueError(f"class_name ({class_name}) is repeated")
        
            class_idx_to_class_name[class_idx] = class_name
            class_name_to_class_idx[class_name] = class_idx
    return class_idx_to_class_name, class_name_to_class_idx


def augment_images(images):
    """
    Augment images during training to maintain generalizability.  This includes
    adding random rotation, flipping, and zoom.
    """
    for layer in image_augmentation_layers:
        images = layer(images)
    return images


def preprocess_images(images):
    """
    Image preprocessing to transform images into the expected format via
    resizing and normalization steps.
    """
    for layer in image_preprocessing_layers:
        images = layer(images)
    return images


def process_image(img_arr):
    """
    Given an image in numpy array format, preprocess it into the form required
    as input to model prediction.  Do this in the same way as done during model
    training.
    """
    img_tensor = tf.convert_to_tensor(img_arr)
    processed_img_arr = preprocess_images(img_tensor).numpy()
    return processed_img_arr


def predict(image_path, model, top_k, verbose=False):
    """
    Predict the model's predicted <top_k> most likely classes and probabilities 
    for the image.
    """
    # load image
    img = Image.open(image_path, "r")
    img_arr = np.asarray(img)

    # process the image and reshape it from (img_height, img_width, num_channels) to
    #  (1, img_height, img_width, num_channels), which is the shape expected by 
    #  model.predict()
    processed_img_arr = process_image(img_arr)
    processed_img_arr = processed_img_arr.reshape((1, *processed_img_arr.shape))

    # generate predictions
    predictions = model.predict(processed_img_arr, verbose=verbose)
    idx = np.argsort(-1.0 * predictions)
    top_k_probs = predictions[0,idx[0,:top_k]]
    top_k_class_idxs = idx[0,:top_k]

    return top_k_probs, top_k_class_idxs


def predict_from_saved_model(args):
    """
    Predict the saved model's predicted <top_k> most likely classes and 
    probabilities for the image.  This is the main function for the
    predict.py script.
    """
    if not os.path.isfile(args.image_file_path):
        raise ValueError(f"image input image file does not exist ({args.image_file_path})")
    if not os.path.isfile(args.model_file_path):
        raise ValueError(f"image input model file does not exist ({args.model_file_path})")

    # load saved model
    model = tf.keras.models.load_model(args.model_file_path)

    # read in dictionary mapping from class_idx to class_name
    class_idx_to_class_name = None
    if args.category_names is not None:
        class_idx_to_class_name, _ = get_class_idx_and_name_mappings_from_json(args.category_names)

    # predict
    top_probs, top_class_idxs = predict(args.image_file_path, model, args.top_k, verbose=False)

    logging.info("class_idx     prob       class_name")
    for class_idx, class_prob in zip(top_class_idxs, top_probs):
        # if class_idx_to_class_name mapping exists, then print the class name
        if class_idx_to_class_name is not None:
            class_name = class_idx_to_class_name[class_idx]
        else:
            class_name = ""
        logging.info(f"{str(class_idx).ljust(14)}{class_prob:.4f}     {class_name}")

    return