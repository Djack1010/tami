from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.metrics import Precision, Recall, AUC


class KhjCNN:
    """
    Model based on the following paper:
     'Kim, Hae-Jung. "Image-based malware classification using convolutional neural network."
     Advances in computer science and ubiquitous computing. Springer, Singapore, 2017. 1352-1357.'
     LINK: (https://link.springer.com/chapter/10.1007/978-981-10-7605-3_215)
    """

    def __init__(self, num_classes, img_size, channels, name="khj"):
        self.name = name
        self.num_classes = num_classes
        self.input_width_height = img_size
        self.channels = channels
        self.input_type = 'images'

    def build(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(self.input_width_height,
                                                                            self.input_width_height,
                                                                            self.channels)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['acc', Precision(name="prec"), Recall(name="rec"), AUC(name='auc')])

        return model
