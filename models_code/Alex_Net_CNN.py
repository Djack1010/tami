from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.metrics import Precision, Recall, AUC


class AlexNet:

    def __init__(self, num_classes, img_size, channels, name="Alex_Net"):
        self.name = name
        self.num_classes = num_classes
        self.input_width_height = img_size
        self.channels = channels
        self.input_type = 'images'

    def build(self):

        model = models.Sequential()
        model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                                input_shape=(self.input_width_height,
                                             self.input_width_height,
                                             self.channels)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'),)
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'), )
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', run_eagerly=True,
                      metrics=['acc', Precision(name="prec"), Recall(name="rec"), AUC(name='auc')])

        return model
