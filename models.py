import tensorflow as tf
import tensorflow_hub as hub


def make_model(num_classes, hidden_units_list=[]):
    """
    Build a transfer learning model using MobileNetV3Large as the base model.  If 
    a list with the number of hidden units is not provided, only the last layer with a
    softmax activation function is added.
    """
    dropout = 0.2
    classifier = tf.keras.applications.MobileNetV3Large(weights="imagenet", include_preprocessing=False, classifier_activation=None)
    classifier.trainable = False

    # add classifier to model
    model = tf.keras.Sequential()
    model.add(classifier)

    # add hidden layers
    for hidden_units in hidden_units_list:
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(hidden_units, activation="relu"))

    # add final layer
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    return model


def make_model_with_preprocessing(num_classes):
    """
    Make a model that includes all the preprocessing steps.  If the dataset has
    images of different sizes, you will need to set batch size to 1.  This is no
    longer used, but is retained as an example.
    """
    # load MobileNet v3 pretrained classifier and freeze weights
    URL = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5"
    feature_extractor = hub.KerasLayer(URL, input_shape=(224, 224, 3))
    feature_extractor.trainable = False

    # image preprocessing steps
    inputs = tf.keras.Input(shape=(None, None, 3))
    x = tf.keras.layers.Resizing(height=256, width=256, crop_to_aspect_ratio=True)(inputs)
    x = tf.keras.layers.CenterCrop(height=224, width=224)(x)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    # pass through pretrained model    
    x = feature_extractor(x)

    # add final layers
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def make_model2(num_classes):
    """
    Alternative approach to make_model().  Kept as an example, but it no longer used.
    """
    img_height, img_width, img_num_channels = 224, 224, 3
    
    # load MobileNet v3 pretrained classifier and freeze weights
    feature_extractor = tf.keras.applications.MobileNetV3Large(weights="imagenet", include_preprocessing=False, 
                                                               classifier_activation=None, dropout_rate=0.25)
    feature_extractor.trainable = False

    # pass through pretrained model    
    inputs = tf.keras.Input(shape=(img_height, img_width, img_num_channels))
    x = feature_extractor(inputs)

    # add a dropout layer
    x = tf.keras.layers.Dropout(0.25)(x)

    # add final layer
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)   

    return tf.keras.Model(inputs, outputs)