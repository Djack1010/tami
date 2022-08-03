from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.regularizers import l1, l1_l2, l2
from tensorflow.keras.optimizers import Adam


class AfaCNNSVM:
    """
    Model based on the following paper:
     'Agarap, Abien Fred. "Towards building an intelligent anti-malware system: a deep learning approach using
     support vector machine (SVM) for malware classification." arXiv preprint arXiv:1801.00318 (2017).'
     LINK: (https://arxiv.org/pdf/1801.00318.pdf)
    """

    def __init__(self, num_classes, img_size, channels, learning_rate=0.01, name="afaCNNSVM"):
        self.name = name
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.input_width_height = img_size
        self.channels = channels
        self.input_type = 'images'

    def build(self):

        model = models.Sequential()
        model.add(layers.Conv2D(36, (5, 5), activation='relu', input_shape=(self.input_width_height,
                                                                            self.input_width_height,
                                                                            self.channels)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(72, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.85))
        model.add(layers.Dense(self.num_classes, kernel_regularizer=l2(0.01), activation='softmax'))

        model.compile(loss='squared_hinge', optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['acc', Precision(name="prec"), Recall(name="rec"), AUC(name='auc')])

        return model
