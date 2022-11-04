from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.metrics import Precision, Recall, AUC


class Inception:

    def __init__(self, num_classes, img_size, channels, weights='imagenet', name="Inception", include_top=False):
        self.name = name
        self.weights = weights
        self.include_top = include_top
        self.num_classes = num_classes
        self.input_width_height = img_size
        self.channels = channels
        self.input_type = 'images'

    def build(self):

        base_model = None
        output = None

        if self.include_top:
            if self.input_width_height != 299 or self.channels != 3:
                print("IF include_top=True, input_shape MUST be (224,224,3), exiting...")
                exit()
            else:
                if self.name == "Inception" or self.name == "InceptionV3":
                    base_model = InceptionV3(weights=self.weights, include_top=False, classes=self.num_classes)
                else:
                    print("Invalid name, accepted 'InceptionV3', exiting...")
                    exit()
                output = base_model.output
        else:
            inputs = Input(shape=(self.input_width_height, self.input_width_height, self.channels))
            if self.name == "Inception" or self.name == "InceptionV3":
                #base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=inputs)
                input_tensor = Input(shape=(299, 299, 3))
                base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
                #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(self.input_width_height, self.input_width_height, self.channels))
            else:
                print("Invalid name, accepted 'Inception', exiting...")
                exit()
            flatten = Flatten(name='my_flatten')
            output_layer = Dense(self.num_classes, activation='softmax', name='my_predictions')
            output = output_layer(flatten(base_model.output))

        input_layer = base_model.input

        model = Model(input_layer, output)
        # model.summary(line_length=50)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['acc', Precision(name="prec"), Recall(name="rec"), AUC(name='auc')])
        return model
