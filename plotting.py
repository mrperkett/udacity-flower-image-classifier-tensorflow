import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import seaborn as sns
from utils import predict, process_image

def plot_image(image, title=None):
    """
    """
    _, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(title, fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_grid_augmented_images(images, image_modification_func, n_rows=4, n_columns=3, rescale_factor=None):
    """
    """
    # TODO: write this function to take a dataset as input and the number of images
    #       to "take" from it as arguments.  It's more complicated to generate a nice
    #       grid of images when they all have different aspect ratios, but it would
    #       make this function more versatile
    # NOTE: modified and improved from the tensorflow tutorial
    #       https://keras.io/examples/vision/image_classification_from_scratch/
    augmented_images = image_modification_func(images)
    height, width, num_channels = augmented_images.shape
    aspect_ratio = width / height
    fig = plt.figure(figsize=(3*n_columns*aspect_ratio, 3*n_rows))
    for i in range(n_rows * n_columns):
        augmented_images = image_modification_func(images)
        if rescale_factor is not None:
            augmented_images *= np.array([rescale_factor], dtype=np.uint8)
        ax = plt.subplot(n_rows, n_columns, i + 1)
        ax.imshow(np.array(augmented_images).astype("uint8"))
        ax.set_axis_off()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    return

def plot_training_metrics(history):
    """
    """
    training_accuracy = history.history["accuracy"]
    validation_accuracy = history.history["val_accuracy"]

    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    _, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].plot(training_accuracy, label="training set")
    ax[0].plot(validation_accuracy, label="validation set")
    ax[0].legend()
    ax[0].set_title("accuracy during training")


    ax[1].plot(training_loss, label="training set")
    ax[1].plot(validation_loss, label="validation set")
    ax[1].legend()
    ax[1].set_title("loss during training")

    return ax


def plot_predictions(image_file_path, model, class_idx_to_class_name, top_k=5):
    """
    Process provided image and display it above the predicted class probability for the topk
    most likely classes.
    """
    # NOTE: this function was modified from my code for another Udacity project
    #       https://github.com/mrperkett/udacity-project-create-image-classifier/tree/main
    if top_k > len(class_idx_to_class_name):
        raise ValueError(f"The top `top_k` ({top_k}) predictions were requested, but there are only {len(class_idx_to_class_name)} classes")

    image_file_name = os.path.basename(image_file_path)

    top_k_probs, top_k_class_idxs = predict(image_file_path, model, top_k=5)
    top_k_class_names = [class_idx_to_class_name[class_idx] for class_idx in top_k_class_idxs]

    # create subplots with one row and two columns.
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.8))

    # display processed flower image
    img = Image.open(image_file_path, "r")
    img_arr = np.asarray(img)
    processed_img_arr = process_image(img_arr)
    ax[0].imshow(np.asarray(processed_img_arr))
    ax[0].set_title(image_file_name)

    # Remove ticks and labels
    # modified from https://stackoverflow.com/questions/12998430/how-to-remove-xticks-from-a-plot
    ax[0].tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    # plot top 5 predicted class probabilities
    _ = sns.barplot(x=top_k_probs, y=top_k_class_names, ax=ax[1])

    # prevent labels from probability bar plot from overlapping image
    plt.tight_layout()

    return ax