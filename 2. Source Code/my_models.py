import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# Defining a mixed Convolutional Neural Network with MLP head
def full_hypermodel_builder(n_classes:int,
                            hp:kt.engine.hyperparameters.HyperParameters,
                            ) -> tf.keras.models.Sequential:
    """
    Define and compile a mixed Convolutional Neural Network with MLP head, 
    which will be used for hyperparameter tuning
    
    :param hp: HyperParameters object
    :return: tf.keras.models.Sequential object
    """
    model = Sequential()

    # Define Convolutional part of the model
    model.add(Conv2D(filters=hp.Int('conv_1_filter', min_value=32, max_value=64, step=16),
                     kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
                     padding=hp.Choice('conv_1_padding', values=['same', 'valid']),
                     kernel_regularizer=l2(hp.Choice('conv_1_l2', values=[0.01, 0.001, 0.0001])),
                     activation='relu', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=32),
                     kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
                     padding=hp.Choice('conv_2_padding', values=['same', 'valid']),
                     kernel_regularizer=l2(hp.Choice('conv_2_l2', values=[0.01, 0.001, 0.0001])),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(filters=hp.Int('conv_3_filter', min_value=64, max_value=256, step=64),
                     kernel_size=hp.Choice('conv_3_kernel', values=[3, 5]),
                     kernel_regularizer=l2(hp.Choice('conv_3_l2', values=[0.01, 0.001, 0.0001])),
                     padding=hp.Choice('conv_3_padding', values=['same', 'valid']),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    # Define MLP part of the model
    model.add(Dense(units=hp.Int('dense_1_units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('dense_2_units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('dense_3_units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(n_classes, activation='softmax'))

    # Compiling the model with Adam optimizer and SparseCategoricalCrossentropy loss function
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])),
                  loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


def tuned_model(compile: bool = True
                   ) -> tf.keras.models.Sequential:
    """
    Define and compile a mixed Convolutional Neural Network with MLP head

    :param compile: bool, whether to compile the model or not
    :return: tf.keras.models.Sequential object
    """
    tuned_model = Sequential([
    # Convolutional layers
    Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(32, 32, 3)),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((2,2)),
    BatchNormalization(),
    Flatten(),

    # Dense layers
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
    ])

    if compile:
        tuned_model.compile(optimizer=Adam(learning_rate=0.001),
                            loss=SparseCategoricalCrossentropy(from_logits=False),
                            metrics=['accuracy'])
        
    return tuned_model


def complex_model(compile: bool = True
                 ) -> tf.keras.models.Sequential:
        """
        Define and compile a complex Convolutional Neural Network with MLP head.
        NOTE: This model is not used in the notebook, but it is provided for your reference,
              if you want to try it out; it could be a good starting point for your experiments,
              but it also could turn out to be not so different from the tuned model, from the
              notebook, in terms of performance.
        
        :param compile: bool, whether to compile the model or not
        :return: tf.keras.models.Sequential object
        """
        complex_model = Sequential([
        # Convolutional layers
        Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
    
        Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
    
        Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
    
        Conv2D(256, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
    
        Conv2D(512, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Flatten(),
    
        # Dense layers
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
        ])
    
        if compile:
            complex_model.compile(optimizer=Adam(learning_rate=0.001),
                                loss=SparseCategoricalCrossentropy(from_logits=False),
                                metrics=['accuracy'])
            
        return complex_model
