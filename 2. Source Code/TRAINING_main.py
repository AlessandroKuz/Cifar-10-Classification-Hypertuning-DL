import numpy as np

import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import cifar_preprocessing
import my_models
import tune
import visualize


def hypertuning(classes: list[str]
               ) -> None:
    """
    Perform hyperparameter tuning.

    :param classes: list[str], the class names
    """
    # Load the scaled train and test sets
    (x_train, y_train), (x_test, y_test) = cifar_preprocessing.load_data(scale_data=True)

    print("Train set shape:", x_train.shape)
    print("Test set shape:", x_test.shape)
    print("Train labels shape:", y_train.shape)
    print("Test labels shape:", y_test.shape)

    # Visualize the data
    visualize.plot_sample_data(x_train, y_train, classes)

    # Convert the data to tensors - UNCOMMENT TO USE -> BETTER PERFORMANCE
    # train_set, test_set = cifar_preprocessing.convert_to_tensors((x_train, y_train), (x_test, y_test))
    
    # Get the optimal hyperparameters for the model
    best_params, hypertuner = tune.tune_model_hyperparameters(my_models.full_hypermodel_builder,
                                                              (x_train, y_train),
                                                              (x_test, y_test))
    
    best_epochs = tune.tune_model_epochs(hypertuner=best_params,
                                         train_set=(x_train, y_train),
                                         test_set=(x_test, y_test),
                                         best_hps=best_params)

    # Build the model with the optimal hyperparameters
    hypermodel = hypertuner.hypermodel.build(best_params)

    # Define model checkpoints 
    hypermodel_checkpoints = ModelCheckpoint(filepath='/hypermodel/hypermodel{epoch}.keras', monitor='val_loss', save_best_only=True),

    # Retrain the model
    hypermodel.fit(x_train, y_train, epochs=best_epochs, validation_split=0.2, callbacks=hypermodel_checkpoints)

    # Evaluate the model on the test set
    eval_result = hypermodel.evaluate(x_test, y_test)

    # Print the test loss and test accuracy
    print("[Test loss, Test accuracy]:", eval_result)

    # Save the tuned model for future reference
    hypermodel.save('hypertuned-cifar-cnn-model.keras')

    # Visualize the model's training history
    visualize.plot_training_history(hypermodel.history, plot_type='complete')

    # Visualize the model's confusion matrix
    visualize.plot_confusion_matrix(hypermodel, (x_test, y_test), classes)


def normal_training(classes: list[str]
                   ) -> None:
    """
    Perform regular training on a model (without hypertuning).

    :param classes: list[str], the class names
    :param n_classes: int, the number of classes
    """
    # Load the scaled train and test sets
    (x_train, y_train), (x_test, y_test) = cifar_preprocessing.load_data(scale_data=True)

    # Convert the data to tensors
    train_set, test_set = cifar_preprocessing.convert_to_tensors((x_train, y_train), (x_test, y_test))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    ]
    
    # Build the model
    tuned_model = my_models.tuned_model(compile=True)
    # OR YOU COULD TRY THE FOLLOWING:
    # tuned_model = my_models.complex_model(compile=True)


    # This time the model uses better callbacks, batching and data shuffling
    tuned_model_history = tuned_model.fit(train_set, batch_size=64, shuffle=True, epochs=50,
                                        validation_split=0.2, callbacks=callbacks)

    # Printing the evaluation metrics

    eval_result = tuned_model.evaluate(test_set)
    print("[Test loss, Test accuracy]:", eval_result)

    # Save the tuned model for future reference
    tuned_model.save('tuned-cifar-cnn-model.keras')

    # Visualize the model's training history
    visualize.plot_training_history(tuned_model_history, plot_type='complete')

    # Visualize the model's confusion matrix
    visualize.plot_confusion_matrix(tuned_model, test_set, classes)



if __name__ == '__main__':
    # Define constants: class names and number 
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Perform hypertuning
    hypertuning(CLASSES)

    
    # UNCOMMENT TO RUN NORMAL TRAINING OF A MODEL ON THE CIFAR-10 DATASET
    # OR Perform normal training, without hypertuning, SAMPLE CODE IS PROVIDED, FEEL FREE TO CHANGE IT
    # normal_training(CLASSES)
