from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
from code_models.gradcams.gradcam_utils import build_guided_model, GuidedBackPropagation
import utils.handle_modes as modes


class GradCAM:
    def __init__(self, model, args, class_info, target_layer_name=None, target_layer_min_shape=1):
        """
        Store the model, the class index used to measure the class activation map,
        and the target layer to be used when visualizing the class activation map.
        Also, target_layer_min_shape can be set to select the target output layer
        with AT LEAST size target_layer_min_shape x target_layer_min_shape size,
        to compare different model on same heatmap resolution
        """

        if '-guided' in args.mode:
            self.model = build_guided_model(modes.load_model, args, class_info)
            self.guided = True
        else:
            self.model = model
            self.guided = False

        # if the layer name is None, attempt to automatically find
        # the target output layer
        if target_layer_name is None:
            # attempt to find the final convolutional layer in the network
            # by looping over the layers of the network in reverse order
            for layer in reversed(self.model.layers):
                # check to see if the layer has a 4D output
                if len(layer.output_shape) == 4 and layer.output_shape[1] >= target_layer_min_shape:
                    self.target_layer_name = layer.name
                    return

            # otherwise, we could not find a 4D layer so the GradCAM algorithm cannot be applied
            raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

        else:
            self.target_layer_name = target_layer_name

    def compute_heatmap(self, image, **kwargs):
        if self.guided:
            return self.compute_guidedSaliency(image)
        else:
            return self._compute_heatmap(image, **kwargs)
    def _compute_heatmap(self, image, **kwargs):

        class_index = kwargs.get('class_index', None)
        eps = kwargs.get('eps', 1e-8)

        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model

        grad_model = Model(inputs=[self.model.inputs], outputs=[self.model.get_layer(self.target_layer_name).output,
                                                                self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = grad_model(inputs)
            loss = predictions[:, class_index]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        output, grads_val = convOutputs[0, :], grads[0, :, :, :]
        weights = np.mean(grads_val, axis=(0, 1))

        heatmap = np.dot(output, weights)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(heatmap, (w, h))

        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def compute_guidedSaliency(self, image):
        return GuidedBackPropagation(self.model, image, self.target_layer_name)
