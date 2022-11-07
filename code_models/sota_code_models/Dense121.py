from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam


class DenseNet:

    def __init__(self, num_classes, img_size, channels, weights='imagenet', learning_rate=0.01, name="DenseNet",
                 include_top=False):
        self.name = name
        self.learning_rate = learning_rate
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
            if self.input_width_height != 224 or self.channels != 3:
                print("IF include_top=True, input_shape MUST be (224,224,3), exiting...")
                exit()
            else:
                if self.name == "DenseNet" or self.name == "DenseNet121":
                    base_model = DenseNet121(weights=self.weights, include_top=False, classes=self.num_classes)
                else:
                    print("Invalid name, accepted 'DenseNet', exiting...")
                    exit()
                output = base_model.output
        else:
            # TODO: check 'inputs' and required size
            inputs = Input(shape=(self.input_width_height, self.input_width_height, self.channels))
            if self.name == "DenseNet" or self.name == "DenseNet121":
                #base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=inputs)
                input_tensor = Input(shape=(224, 224, 3))
                base_model = DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False)
                #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(self.input_width_height, self.input_width_height, self.channels))
            else:
                print("Invalid name, accepted 'DenseNet', exiting...")
                exit()
            flatten = Flatten(name='my_flatten')
            output_layer = Dense(self.num_classes, activation='softmax', name='my_predictions')
            output = output_layer(flatten(base_model.output))

        input_layer = base_model.input

        model = Model(input_layer, output)
        # model.summary(line_length=50)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(self.learning_rate),
                      metrics=['acc', Precision(name="prec"), Recall(name="rec"), AUC(name='auc')])
        return model
