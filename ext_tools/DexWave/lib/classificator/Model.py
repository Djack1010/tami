#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

class MCModel(tf.keras.Model):
    def __init__(self, num_classes, img_size, channels):
        super(MCModel, self).__init__()
        # Variables
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels
        self.trained = False

        #Model Layers
        self.layer_input = tf.keras.layers.InputLayer(input_shape=(self.img_size, self.img_size, self.channels))
        self.layer_conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.layer_maxpooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.layer_conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.layer_maxpooling2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.layer_flatten = tf.keras.layers.Flatten()
        self.layer_dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.layer_dropout1 = tf.keras.layers.Dropout(0.5)
        self.layer_dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.layer_dropout2 = tf.keras.layers.Dropout(0.5)
        self.layer_dense_out = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=None):
        x = self.layer_input(x)
        x = self.layer_conv1(x)
        x = self.layer_maxpooling1(x)
        x = self.layer_conv2(x)
        x = self.layer_maxpooling2(x)
        x = self.layer_flatten(x)
        x = self.layer_dense1(x)
        x = self.layer_dropout1(x, training=training)
        x = self.layer_dense2(x)
        x = self.layer_dropout2(x, training=training)
        return self.layer_dense_out(x)

