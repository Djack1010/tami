from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import Precision, Recall, AUC


class ResNet:

    def __init__(self, num_classes, img_size, channels, weights='imagenet', name="ResNet50", include_top=False):
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
            if self.input_width_height != 224 or self.channels != 3:
                print("IF include_top=True, input_shape MUST be (224,224,3), exiting...")
                exit()
            else:
                if self.name == "ResNet50" or self.name == "ResNet":
                    base_model = ResNet50(weights=self.weights, include_top=False, classes=self.num_classes)
                else:
                    print("Invalid name, accepted 'InceptionV3', exiting...")
                    exit()
                output = base_model.output
        else:
            inputs = Input(shape=(self.input_width_height, self.input_width_height, self.channels))
            if self.name == "ResNet50" or self.name == "ResNet":
                base_model = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
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
