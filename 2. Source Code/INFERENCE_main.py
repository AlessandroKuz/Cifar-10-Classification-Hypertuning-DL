import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

import cifar_preprocessing
import visualize


if __name__ == '__main__':
    
    # Path suggestion: r'C:\Users\AlessandroKuz\Documents\GitHub\Cifar-10-Classification-Hypertuning-DL\Models\...'
    MODEL_PATH = "insert your model path here"


    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog','frog', 'horse', 'ship', 'truck']

    # Load the scaled test set
    _, (x_test, y_test) = cifar_preprocessing.load_data(scale_data=True)

    # UNCOMMENT TO VISUALIZE THE TEST IMAGES
    # visualize.plot_sample_data(x_test, y_test, CLASSES)

    # Load the model
    model = load_model(MODEL_PATH)
    # Evaluate the model on the test set
    eval_result = model.evaluate(x_test, y_test)

    # Print the test loss and test accuracy
    print("[Test loss, Test accuracy]:", eval_result)

    predictions = model.predict(x_test[:4])
    print(predictions)

    # Plot the first 4 test images, their predicted labels, and the true labels using matplotlib
    fig = plt.figure(figsize=(10, 10))
    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_test[i]))
        pred_idx = np.argmax(predictions[i])
        true_idx = np.argmax(y_test[i])
        ax.set_title("{} ({})".format(CLASSES[pred_idx], CLASSES[true_idx]),
                     color=("green" if pred_idx == true_idx else "red"))

    # Plot the confusion matrix
    visualize.plot_confusion_matrix(model, (x_test, y_test), CLASSES, plot_misclassified_images=True)
