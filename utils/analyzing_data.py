import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import io
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


def plot_training_result(dict_res, save_on_file=None):
    # lets plot the train and val curve
    # get the details form the history object
    acc = dict_res['acc']
    val_acc = dict_res['val_acc']
    loss = dict_res['loss']
    val_loss = dict_res['val_loss']

    epochs = range(1, len(acc) + 1)

    # Train and validation accuracy
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xticks(epochs)
    plt.title('Training and Validation accurarcy')
    plt.legend()

    if save_on_file is not None:
        plt.savefig(save_on_file + "_trainAcc.png")

    plt.figure()
    # Train and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xticks(epochs)
    plt.title('Training and Validation loss')
    plt.legend()

    if save_on_file is None:
        plt.show()
    else:
        plt.savefig(save_on_file + "_trainLoss.png")


def cm_4_exp(model_class, test_ds, fw_cm):
    # TODO Improve efficiency
    print("Generating Confusion Matrix...    ", end="\r")
    model = model_class.model
    complete_test = test_ds.unbatch()
    test_pred = []
    y_test_int = []
    for images, labels in complete_test.take(-1):
        y_test_int.append(np.argmax(labels.numpy(), axis=0))
        test_pred.append(np.argmax(model(np.reshape(images.numpy(), (-1, model_class.input_width_height,
                                                                     model_class.input_width_height,
                                                                     model_class.channels)),
                                         training=False).numpy(), axis=1)[0])
    log_confusion_matrix(test_pred, y_test_int, fw_cm[0], fw_cm[1])

def plot_confusion_matrix(cm, class_names, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = cm.round(decimals=2)

    figure = plt.figure(figsize=(30, 25))
    df_cm = pd.DataFrame(cm)  # , index=class_names, columns=class_names
    sn.set(font_scale=4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 60}, cmap=plt.cm.Blues, fmt='g')  # font size
    plt.title(title)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, horizontalalignment='center', rotation=30)
    plt.yticks(tick_marks, class_names, verticalalignment='top', rotation=30)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def log_confusion_matrix(pred, y, file_writer_cm, class_names, epoch=0):
    # Calculate the confusion matrix and log the confusion matrix as an image summary.
    figure_norm = plot_confusion_matrix(confusion_matrix(pred, y, normalize='true'), class_names=class_names,
                                        title="Confusion Matrix Normalized")
    cm_image_norm = plot_to_image(figure_norm)
    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix Normalized", cm_image_norm, step=epoch)
