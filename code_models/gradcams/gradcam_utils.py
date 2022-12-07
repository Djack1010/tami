import cv2
import numpy as np

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