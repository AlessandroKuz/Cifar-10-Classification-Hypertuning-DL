import numpy as np
import tensorflow as tf
import keras_tuner as kt


def tune_model_hyperparameters(model: tf.keras.models.Sequntial,
                               train_set: tf.data.Dataset | tuple(np.ndarray, np.ndarray),
                               test_set: tf.data.Dataset | tuple(np.ndarray, np.ndarray) | None = None,
                               directory: str = 'cifar-10_hypertuning',
                               project_name: str = 'cnn_cifar_10'
                               ) -> (dict, kt.Hyperband):
    """
    Tune the hyperparameters of the model.

    :param model: tf.keras.models.Sequential, the model to tune
    :param train_set: tf.data.Dataset | tuple(np.ndarray, np.ndarray), the training set
    :param test_set: tf.data.Dataset | tuple(np.ndarray, np.ndarray), the test set
    :param directory: str, the directory to save the hypertuning results
    :param project_name: str, the name of the project
    :return: dict, the best hyperparameters & kt.Hyperband, the hypertuner object
    """
                             
    # Instantiate the tuner - using Hyperband algorithm
    hypertuner = kt.Hyperband(model,
                        objective='val_accuracy',
                        max_epochs=10,
                        factor=3,
                        directory=directory,
                        project_name=project_name)

    # Create a callback to stop training early if no progress is made
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    try:
        if isinstance(train_set, tuple):
            x_train, y_train = train_set
            hypertuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stop])
        else:
            hypertuner.search(train_set, epochs=50, validation_data=test_set, callbacks=[early_stop])
    except Exception as e:
        print(e)
        print("Error while tuning the model. Please check that input data is of the same type. Exiting...")
        exit(1)
    # Get the optimal hyperparameters
    best_hps = hypertuner.get_best_hyperparameters(num_trials=1)[0]

    return best_hps.values, hypertuner


def tune_model_epochs(hypertuner: kt.Hyperband,
                      train_set: tf.data.Dataset | tuple(np.ndarray, np.ndarray),
                      test_set: tuple(np.ndarray, np.ndarray),
                      best_hps: dict,
                      max_epochs: int = 50,
                      return_history: bool = False
                     ) -> int:
    """
    Tune the number of epochs of the model.

    :param model: tf.keras.models.Sequential, the model to tune
    :param hypertuner: kt.Hyperband, the hypertuner object
    :param x_train: numpy array, training set
    :param y_train: numpy array, training labels
    :param x_test: numpy array, testing set
    :param y_test: numpy array, testing labels
    :param best_hps: dict, the best hyperparameters
    :return: int, the best number of epochs for the model
    """
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    tuned_model = hypertuner.hypermodel.build(best_hps)
    
    try:
        if isinstance(train_set, tuple):
            x_train, y_train = train_set
            history = tuned_model.fit(x_train, y_train, validation_split=0.2)
        else:
            history = tuned_model.fit(train_set, epochs=max_epochs, validation_data=test_set)
    except Exception as e:
        print(e)
        print("Error while tuning the model. Please check that input data is of the same type. Exiting...")
        exit(1)

    # Getting the best epoch - for future reproducibility
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    
    return best_epoch, history if return_history else best_epoch
