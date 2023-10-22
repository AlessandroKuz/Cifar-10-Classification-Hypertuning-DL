import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_sample_data(x_data: np.ndarray,
                     y_data: np.ndarray,
                     classes_names: list
                    ) -> None:
    """
    Plot 9 images from the dataset.

    :param x_data: numpy array, data images
    :param y_data: numpy array, data labels
    :param classes_names: list, the class names
    :return: None
    """
    # Define the dimensions of the plot grid
    plt.figure(figsize=(10, 10))
    for i in range(9):
        # Define the subplot (region of the plot grid)
        _ = plt.subplot(3, 3, i + 1)
        image = x_data[i]
        label_index = y_data[i][0]
        label = classes_names[label_index]
        # Plot the image with its label
        plt.imshow(image)
        plt.title(label)
        plt.axis("off")  # Remove the axes for cleaner display
    plt.show()


def plot_training_history(history: tf.keras.callbacks.History,
                          plot_type: str = 'loss'
                          ) -> None:
    """
    Plot the training history.

    :param history: tf.keras.callbacks.History, the training history
    :param plot_type: str, the type of plot to display, either 'loss', 'accuracy' or 'complete'
    :return: None
    """
    if plot_type.lower() != 'complete':
        plot_history_type(history, plot_type)
    else:
        plot_history_type(history, 'loss')
        plot_history_type(history, 'accuracy')


def plot_history_type(history: tf.keras.callbacks.History,
                      plot_type: str = 'loss'
                      ) -> None:
    """
    Plot the training history.

    :param history: tf.keras.callbacks.History, the training history
    :param plot_type: str, the type of plot to display, either 'loss' or 'accuracy'
    :return: None
    """
    plt.figure(figsize=(10, 10))

    # Plot the training and validation loss
    plt.plot(history.history[plot_type])
    plt.plot(history.history['val_' + plot_type])

    # Define the plot labels
    plt.title('Training and validation ' + plot_type)
    plt.ylabel(plot_type)
    plt.xlabel('Epoch')
    plt.legend(['Training ' + plot_type, 'Validation ' + plot_type], loc='upper right')

    # Show the plot
    plt.show()


def plot_confusion_matrix(model: tf.keras.models.Sequential,
                          test_set: tf.data.Dataset | tuple(np.ndarray, np.ndarray),
                          class_names: list,
                          plot_misclassified_images: bool = False
                         ) -> None:
    """
    Plot the confusion matrix.

    :param model: tf.keras.models.Sequential, the model
    :param test_set: tf.data.Dataset | tuple(np.ndarray, np.ndarray), the test set, contains y_true
    :param class_names: list, the class names
    :param plot_misclassified_images: bool, whether to also plot (mis)classified images or not
    :return: None
    """
    if isinstance(test_set, tuple):
        x_test, y_test = test_set
        y_true = y_test
        y_pred = model.predict(x_test, y_test)
    else:
        y_true = test_set.map(lambda x, y: y)
        y_pred = model.predict(test_set)

    y_pred_classes = np.argmax(y_pred, axis=1)

    # Display the confusion matrix
    sklearn_confusion_mtx = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sklearn_confusion_mtx, annot=True, fmt='d')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    if plot_misclassified_images:
        # Display some misclassified images
        misclassified_idx = np.where(y_pred_classes != y_true)[0]
        i = np.random.choice(misclassified_idx)
        plt.imshow(x_test[i])
        plt.title(f"True label: {class_names[y_true[i][0]]}\nPredicted label: {class_names[y_pred_classes[i]]}")
        plt.axis('off')
        plt.show()

        # Display some correctly classified images
        correctly_classified_idx = np.where(y_pred_classes == y_true)[0]
        i = np.random.choice(correctly_classified_idx)
        plt.imshow(x_test[i])
        plt.title(f"True label: {class_names[y_true[i][0]]}\nPredicted label: {class_names[y_pred_classes[i]]}")
        plt.axis('off')
        plt.show()
