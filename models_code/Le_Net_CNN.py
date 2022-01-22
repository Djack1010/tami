from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam


class LeNet:

    def __init__(self, num_classes, img_size, channels, learning_rate=0.01, name="Le_Net"):
        self.learning_rate = learning_rate
        self.name = name
        self.num_classes = num_classes
        self.input_width_height = img_size
        self.channels = channels
        self.input_type = 'images'

    def build(self):

        model = models.Sequential()
        model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=(self.input_width_height,
                                                                            self.input_width_height,
                                                                            self.channels)))
        model.add(layers.AveragePooling2D(2))
        model.add(layers.Activation('sigmoid'))
        model.add(layers.Conv2D(16, 5, activation='tanh'))
        model.add(layers.AveragePooling2D(2))
        model.add(layers.Activation('sigmoid'))
        model.add(layers.Conv2D(120, 5, activation='tanh'))
        model.add(layers.Flatten())
        model.add(layers.Dense(84, activation='tanh'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(self.learning_rate),
                      metrics=['acc', Precision(name="prec"), Recall(name="rec"), AUC(name='auc')])

        return model
