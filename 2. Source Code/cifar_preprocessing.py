import numpy as np
import tensorflow as tf


def load_data(scale_data: bool = True
             ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Load data from tensorflow.keras.datasets.cifar10 and optionally scale it.
    
    :param scale_data: bool, whether to scale the data
    :return: tuple[tuple[np.ndarray, np.ndarray],
                  tuple[np.ndarray, np.ndarray]], the training set and the test set
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    if scale_data:
        x_train = x_train / 255.
        x_test = x_test / 255.
    
    return (x_train, y_train), (x_test, y_test)
    

def convert_to_tensors(trainset: tuple(np.ndarray, np.ndarray),
                       testset: tuple(np.ndarray, np.ndarray),
                       batch_size: int = 64
                       ) -> tuple(tf.data.Dataset, tf.data.Dataset):
    """
    Convert the data to tensors.

    :param trainset: tuple(np.ndarray, np.ndarray), the training set
    :param testset: tuple(np.ndarray, np.ndarray), the test set
    :param batch_size: int, the batch size
    :return: tuple(tf.data.Dataset, tf.data.Dataset), the training set and the test set
    """
    AUTOTUNE = tf.data.AUTOTUNE

    # convert the data to use the .cache() method to cache datasets in memory for better performance
    train = tf.data.Dataset.from_tensor_slices(trainset)
    test = tf.data.Dataset.from_tensor_slices(testset)

    train = (
        train
        .shuffle(1000)
        .batch(batch_size)
        .cache()
        .prefetch(AUTOTUNE)
    )

    test = (
        test
        .batch(batch_size)
        .cache()
        .prefetch(AUTOTUNE)
    )

