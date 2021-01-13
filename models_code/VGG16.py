from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg16


class VGG16_19:

    def __init__(self, num_classes, img_size, channels, weights='imagenet', name="VGG", include_top=False):
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
                if self.name == "VGG" or self.name == "VGG16":
                    base_model = vgg16.VGG16(weights=self.weights, include_top=True, classes=self.num_classes)
                else:
                    print("Invalid name, accepted 'VGG1619', exiting...")
                    exit()
                output = base_model.output
        else:
            inputs = Input(shape=(self.input_width_height, self.input_width_height, self.channels))
            if self.name == "VGG" or self.name == "VGG16":
                base_model = vgg16.VGG16(weights=self.weights, include_top=False, input_tensor=inputs)
            else:
                print("Invalid name, accepted 'VGG16', exiting...")
                exit()
            flatten = Flatten(name='my_flatten')
            output_layer = Dense(self.num_classes, activation='softmax', name='my_predictions')
            output = output_layer(flatten(base_model.output))

        input_layer = base_model.input

        model = Model(input_layer, output)
        # model.summary(line_length=50)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model
