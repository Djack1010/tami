import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras import backend as K

def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

def overlay_heatmap(heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS, emphasize=False):
    # apply the supplied color map to the heatmap and then
    # overlay the heatmap on the input image
    heatmap = cv2.applyColorMap(heatmap, colormap)
    if emphasize:
        heatmap = sigmoid(heatmap, 50, 0.5, 1)
    output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

    # return a 2-tuple of the color mapped heatmap and the output,
    # overlaid image
    return heatmap, output

def build_guided_model(build_model_function, args, class_info):
    """Function returning modified model.

    Changes gradient function for all ReLu activations according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model_function(args, class_info['n_classes'])
    return new_model

def GuidedBackPropagation(model, img_array, layer_name):
    model_input = model.input
    layer_output = model.get_layer(layer_name).output
    max_output = K.max(layer_output, axis=3)
    grads = tf.gradients(max_output, model_input)[0]
    get_output = K.function([model_input], [grads])
    saliency = get_output([img_array])
    saliency = np.clip(saliency[0][0], 0.0, 1.0)  # scale 0 to 1.0
    saliency = (saliency * 255).astype("uint8")
    return saliency