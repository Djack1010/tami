from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.regularizers import l1, l1_l2, l2


class AfaMLPSVM:
    """
    Model based on the following paper:
     'Agarap, Abien Fred. "Towards building an intelligent anti-malware system: a deep learning approach using
     support vector machine (SVM) for malware classification." arXiv preprint arXiv:1801.00318 (2017).'
     LINK: (https://arxiv.org/pdf/1801.00318.pdf)
    """

    def __init__(self, num_classes, vector_size, learning_rate=0.01, name="afaMLPSVM"):
        self.name = name
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.vector_size = vector_size
        self.input_type = 'vectors'

    def build(self):

        model = models.Sequential()
        model.add(layers.Dense(512, input_shape=(self.vector_size,), activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.num_classes, kernel_regularizer=l2(0.01), activation='softmax'))

        model.compile(loss='squared_hinge', optimizer='adam',
                      metrics=['acc', Precision(name="prec"), Recall(name="rec"), AUC(name='auc')])

        return model
