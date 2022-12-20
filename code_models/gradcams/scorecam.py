from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import cv2
from code_models.gradcams.gradcam_utils import build_guided_model, GuidedBackPropagation
import utils.handle_modes as modes


class GradCAM:
    def __init__(self, model, args, class_info, target_layer_name=None, target_layer_min_shape=1, max_N=-1):
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

        self.max_N = max_N

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

        act_map_array = Model(inputs=self.model.input,
                              outputs=self.model.get_layer(self.target_layer_name).output).predict(image, steps=1)

        # extract effective maps
        if self.max_N != -1:
            act_map_std_list = [np.std(act_map_array[0, :, :, k]) for k in range(act_map_array.shape[3])]
            unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), self.max_N)[:self.max_N]
            max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
            act_map_array = act_map_array[:, :, :, max_N_indices]

        input_shape = tuple(image.shape[1:]) if len(image.shape) > 3 else tuple(image.shape)

        # 1. upsample to original input size
        act_map_resized_list = [cv2.resize(act_map_array[0, :, :, k], input_shape[:2], interpolation=cv2.INTER_LINEAR)
                                for k in range(act_map_array.shape[3])]
        # 2. normalize the raw activation value in each activation map into [0, 1]
        act_map_normalized_list = []
        for act_map_resized in act_map_resized_list:
            if np.max(act_map_resized) - np.min(act_map_resized) != 0:
                act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
            else:
                act_map_normalized = act_map_resized
            act_map_normalized_list.append(act_map_normalized)
        # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
        masked_input_list = []
        for act_map_normalized in act_map_normalized_list:
            masked_input = np.copy(image)
            for k in range(input_shape[2]):
                masked_input[0, :, :, k] *= act_map_normalized
            masked_input_list.append(masked_input)
        masked_input_array = np.concatenate(masked_input_list, axis=0)
        # 4. feed masked inputs into CNN model and softmax
        pred_from_masked_input_array = np.exp(self.model.predict(masked_input_array)) / \
                                       np.sum(np.exp(self.model.predict(masked_input_array)), axis=1, keepdims=True)

        # 5. define weight as the score of target class
        weights = pred_from_masked_input_array[:, cls]
        # 6. get final class discriminative localization map as linear weighted combination of all activation maps
        heatmap = np.dot(act_map_array[0, :, :, :], weights)
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
