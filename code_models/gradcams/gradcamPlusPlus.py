from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
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

        # class index
        cls = kwargs.get('class_index', None)
        eps = kwargs.get('eps', 1e-8)

        y_c = self.model.output[0, cls]
        conv_output = self.model.get_layer(self.target_layer_name).output
        grads = tf.gradients(y_c, conv_output)[0]

        first = K.exp(y_c) * grads
        second = K.exp(y_c) * grads * grads
        third = K.exp(y_c) * grads * grads * grads

        gradient_function = K.function([self.model.inputs], [y_c, first, second, third, conv_output, grads])
        y_c, conv_first_grad, conv_second_grad, conv_third_grad, conv_output, grads_val = gradient_function(image)
        global_sum = np.sum(conv_output[0].reshape((-1, conv_first_grad[0].shape[2])), axis=0)

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape(
            (1, 1, conv_first_grad[0].shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num / alpha_denom

        weights = np.maximum(conv_first_grad[0], 0.0)
        alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)
        alphas /= alpha_normalization_constant.reshape((1, 1, conv_first_grad[0].shape[2]))
        deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad[0].shape[2])), axis=0)

        heatmap = np.sum(deep_linearization_weights*conv_output[0], axis=2)
        heatmap = np.maximum(heatmap, 0)  # Passing through ReLU

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
