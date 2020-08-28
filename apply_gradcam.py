# USAGE
# python apply_gradcam.py --image images/space_shuttle.jpg
# python apply_gradcam.py --image images/beagle.jpg
# python apply_gradcam.py --image images/soccer_ball.jpg --model resnet

# import the necessary packages
import math

from models.gradcam import GradCAM
from utils.config import *
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import numpy as np
import argparse
import pickle
import imutils
import random
import cv2


def _model_selection(arguments):
    print("LOADING MODEL")
    model = load_model(main_path + 'model_saved/{}_m{}'.format(arguments.load_model, arguments.model))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    with open(main_path + 'model_saved/{}_m{}.info'.format(arguments.load_model, arguments.model), 'rb') \
            as filehandle:
        stored_data = pickle.load(filehandle)
        CN = stored_data["CLASS_NAMES"]
        C = stored_data["CHANNELS"]
        ID = stored_data["IMG_DIM"]
    return model, CN, C, ID


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    # cast to float32 for one_hot encode (otherwise TRUE/FALSE tensor)
    return tf.cast(parts[-2] == CLASS_NAMES, tf.float32)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=CHANNELS)  # tf.image.decode_jpeg(img, channels=CHANNELS)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_DIM, IMG_DIM])


def process_path(file_path, tensor=True):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    if tensor:
        img = decode_img(img)
    return img, label


def merg_average_pic(pic1, pic2, shape):
    pic_new = np.zeros(shape=(shape[0], shape[1]), dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if len(shape) == 2:
                try:
                    pic_new[i, j] = (int(pic1[i, j]) + int(pic2[i, j])) / 2
                except RuntimeWarning:
                    print("CATCHED: {} {}".format(pic1[i, j], pic2[i, j]))
            else:
                for el in range(shape[2]):
                    pic_new[i, j, el] = (pic1[i, j, el] + pic2[i, j, el]) / 2
    return pic_new


def merg_pics(list_of_pic):
    shape = list_of_pic[0].shape
    if len(shape) != 2:
        print("ERRORE")
        exit()
    pic_new = np.zeros(shape=(shape[0], shape[1]), dtype='uint8')
    pic_std = np.zeros(shape=(shape[0], shape[1]), dtype='uint8')
    # Per each pixel i,j and per each image n, sum up the values and then...
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp = 0
            for n in range(len(list_of_pic)):
                temp += list_of_pic[n][i, j]
            # diveded by the number of pic, to get an AVG pixel
            pic_new[i, j] = temp / len(list_of_pic)
    # Per each pixel i,j and per each image n, sum up the difference between the average (what is inside pic_new[i,j])
    # and the pixel[i,j] of image n (to calculate the distance beetween that pixed to the average).
    # then to the power of 2, divided by number of pics and squared to apply the formula for Standard Deviation
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp = 0
            for n in range(len(list_of_pic)):
                temp += (list_of_pic[n][i, j] - int(pic_new[i, j])) ** 2
            pic_std[i, j] = math.sqrt(temp / len(list_of_pic))
    return pic_new, pic_std


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-m", "--model", type=str, required=True, default="BASIC_EXP", help="model to be loaded")
ap.add_argument("-l", "--load_model", type=str, required=True, default="nedo", help="model name to be loaded")
ap.add_argument("-d", "--dataset", type=str, required=True, help="database preprocessed with samples "
                                                                 "for generating images")
arguments = ap.parse_args()

# Perform operation on CPU to avoid loading of libraries for GPU (no training required)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# initialize the model
model, CLASS_NAMES, CHANNELS, IMG_DIM = _model_selection(arguments)
# model.summary(line_length=50)

# initialize our gradient class activation map
cam = GradCAM(model)

if not os.path.isdir(main_path + arguments.dataset) or "_PREPROCESSED" not in main_path + arguments.dataset:
    print('Cannot find dataset preprocessed in {}, exiting...'.format(arguments.dataset))
    exit()

dataset_base = main_path + arguments.dataset

# os.walk(directories) gen a 3-tuple for subdir, dir and files
class_folders = [x[0] for x in os.walk(dataset_base + "/test")]
for f in class_folders:

    if f == dataset_base + "/test":
        continue

    files = [i[2] for i in os.walk(f)]
    num_samples = 50
    imgs = random.sample(files[0], num_samples)
    gray_heatmaps = []
    gray_heatmaps_WRONG = []
    color_heatmaps = []
    fixed_size_studyMap = 700

    img_class = f.split("/")[-1]
    # create folder in /results/images for this class
    if not os.path.isdir(main_path + 'results/images/' + img_class):
        os.mkdir(main_path + 'results/images/' + img_class)

    result_images_path = main_path + 'results/images/' + img_class

    for i in range(num_samples):
        complete_path = f + "/" + imgs[i]
        img_filename = imgs[i].split(".")[0]

        # load the original image from disk (in OpenCV format) and then
        # resize the image to its target dimensions
        orig = cv2.imread(complete_path)
        # resized = cv2.resize(orig, (arguments.image_size, arguments.image_size))

        image, _ = process_path(complete_path)
        image = tf.expand_dims(image, 0)

        # use the network to make predictions on the input imag and find
        # the class label index with the largest corresponding probability
        preds = model.predict(image)
        i = np.argmax(preds[0])

        # decode the ImageNet predictions to obtain the human-readable label
        # decoded = imagenet_utils.decode_predictions(preds)
        # (imagenetID, label, prob) = decoded[0][0]
        # label = "{}: {:.2f}%".format(label, prob * 100)
        correctness = "WRONG " if img_class != CLASS_NAMES[int(i)] else ""
        label = "{}{} - {:.1f}%".format(correctness, CLASS_NAMES[int(i)], preds[0][i] * 100)
        print("[INFO] {}".format(label))

        # build the heatmap
        heatmap = cam.compute_heatmap(image, i)

        # resize to fixed size and add the heatmap to the study struct
        # at this point the heatmap contains integer value scaled [0, 255]
        heatmap_raw = heatmap.copy()
        heatmap_raw = cv2.resize(heatmap_raw, (fixed_size_studyMap, fixed_size_studyMap))
        if correctness == "":
            gray_heatmaps.append(heatmap_raw)
        else:
            gray_heatmaps_WRONG.append(heatmap_raw)

        # resize the resulting heatmap to the original input image dimensions
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

        # overlay heatmap on top of the image
        (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

        # heatmap_comparison for printing also heatmap alone with filename
        heatmap_comparison = heatmap.copy()

        # resize images
        orig = imutils.resize(orig, width=400)
        heatmap = imutils.resize(heatmap, width=400)
        heatmap_comparison = imutils.resize(heatmap_comparison, width=400)
        output = imutils.resize(output, width=400)

        # create a black background to include text
        black = np.zeros((35, orig.shape[1], 3), np.uint8)
        black[:] = (0, 0, 0)

        # concatenate vertically to the image
        orig = cv2.vconcat((black, orig))
        heatmap = cv2.vconcat((black, heatmap))
        heatmap_comparison = cv2.vconcat((black, heatmap_comparison))
        output = cv2.vconcat((black, output))

        # write some text over each image
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
        cv2.putText(heatmap, "Heatmap", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
        cv2.putText(heatmap_comparison, img_filename.split('_')[2], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255))
        cv2.putText(output, "Overlay with Heatmap", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))

        # display the original image and resulting heatmap and output image
        complete = np.hstack([orig, heatmap, output])
        # complete = imutils.resize(complete, width=700)
        semi_complete = np.hstack([orig, output])
        # semi_complete = imutils.resize(semi_complete, width=350)
        cv2.imwrite(result_images_path + '/complete_' + img_filename.split('_')[2] + '.png', complete)
        # cv2.imwrite(result_images_path + '/semi_' + img_filename.split('_')[2] + '.png', semi_complete)

        color_heatmaps.append(heatmap_comparison)

    # Display images
    # cv2.imshow("Original", orig)
    # cv2.imshow("Heatmap", heatmap)
    # cv2.imshow("Overlay", output)
    # cv2.imshow("Complete", complete)
    # cv2.imshow("Semi-Complete", semi_complete)
    # cv2.waitKey(0)

    # for x in range(heatmap_new.shape[0]):
    #	for y in range(heatmap_new.shape[1]):
    #		if sum(heatmap_new[x, y]) < 325:
    #			heatmap_new[x, y] = np.array([0, 0, 0])

    if num_samples >= 5:
        valid_heatmap = []

        for i in range(num_samples):
            if color_heatmaps[i].shape == (435, 400, 3) and not np.all((color_heatmaps[i] == 0)):
                valid_heatmap.append(i)
            if len(valid_heatmap) == 5:
                break

        if len(valid_heatmap) == 5:
            compared_heatmaps = np.hstack([color_heatmaps[valid_heatmap[0]], color_heatmaps[valid_heatmap[1]],
                                           color_heatmaps[valid_heatmap[2]], color_heatmaps[valid_heatmap[3]],
                                           color_heatmaps[valid_heatmap[4]]])
            cv2.imwrite(result_images_path + '/comparison_' + img_class + '.png', compared_heatmaps)

    # for n in range(num_samples):
    #	cv2.imwrite(main_path + 'results/images/gray_' + img_class + '_' + str(n) + '.png', gray_heatmaps[n])
    # pic1 = merg_average_pic(gray_heatmaps[0], gray_heatmaps[1], gray_heatmaps[1].shape)
    # cv2.imwrite(main_path + 'results/images/graysum1_' + img_class + '.png', pic1)
    # pic2 = merg_average_pic(gray_heatmaps[2], gray_heatmaps[3], gray_heatmaps[2].shape)
    # cv2.imwrite(main_path + 'results/images/graysum2_' + img_class + '.png', pic2)
    # pic3 = merg_average_pic(pic1, pic2, pic1.shape)
    # for x in range(pic1.shape[0]):
    #	print(pic3[x])

    # cv2.imwrite(main_path + 'results/images/graysum3_' + img_class + '.png', pic3)

    # merging heatmaps into one cumulative heatmap, creating also standard deviation image
    print("[INFO] Generating Cumulative Heatmap for {}...".format(img_class), end='', flush=True)
    pic_avg, pic_std = merg_pics(gray_heatmaps)
    print("DONE!")
    cv2.imwrite(result_images_path + '/grayscaleAVG_' + img_class + '.png', pic_avg)
    cv2.imwrite(result_images_path + '/grayscaleSTD_' + img_class + '.png', pic_std)

    # add color to cumulative heatmap and STD image
    pic_avg_colored = cv2.applyColorMap(pic_avg, cv2.COLORMAP_VIRIDIS)
    pic_std_colored = cv2.applyColorMap(pic_std, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(result_images_path + '/colorAVG_' + img_class + '.png', pic_avg_colored)
    cv2.imwrite(result_images_path + '/images/colorSTD_' + img_class + '.png', pic_std_colored)

    # Store the raw heatmaps per family
    for ind in range(len(gray_heatmaps)):
        cv2.imwrite(result_images_path + '/heatmap_' + str(ind) + '.png', gray_heatmaps[ind])
    j = len(gray_heatmaps)
    for ind in range(len(gray_heatmaps)):
        cv2.imwrite(result_images_path + '/heatmapWRONG_' + str(j) + '.png', gray_heatmaps[ind])
        j += 1

# n_box = 50
# step = fixed_size_studyMap / n_box
# box = np.zeros(shape=(n_box, n_box), dtype=int)
# box_counter = np.zeros(shape=(n_box, n_box), dtype=int)

# xs = -1
# for x in range(fixed_size_studyMap):
#	if x % step == 0:
#		xs += 1
#	ys = -1
#	for y in range(fixed_size_studyMap):
#		if y % step == 0:
#			ys += 1
#		box[xs, ys] += pic3[x, y]

# pixel_per_box = (fixed_size_studyMap / n_box) * (fixed_size_studyMap / n_box)
# print(box)
# print(box // pixel_per_box)

# Display images
# cv2.imshow("ONE", compl_heatmap[0])
# cv2.imshow("TWO", compl_heatmap[1])
# cv2.imshow("THREE", compl_heatmap[2])

# (heatmap_merg_1, output_merg_1) = cam.overlay_heatmap(compl_heatmap[0], compl_heatmap[1], alpha=0.4)
# cv2.imshow("MERGED_1", output_merg_1)
# (heatmap_merg_2, output_merg_2) = cam.overlay_heatmap(compl_heatmap[2], compl_heatmap[3], alpha=0.4)
# (heatmap_merg_3, output_merg_3) = cam.overlay_heatmap(output_merg_1, output_merg_2, alpha=0.4)
# cv2.imshow("MERGED_2", output_merg_2)
# semi_complete_merged1 = np.hstack([compl_heatmap[0], compl_heatmap[1], compl_heatmap[2], compl_heatmap[3]])
# semi_complete_merged1 = imutils.resize(semi_complete_merged1, height=450)
# cv2.imshow("BASELINE", semi_complete_merged1)
# semi_complete_merged2 = np.hstack([output_merg_1, output_merg_2, output_merg_3])
# semi_complete_merged2 = imutils.resize(semi_complete_merged2, height=400)
# cv2.imshow("MERGED", semi_complete_merged2)
# cv2.imwrite(main_path + 'results/images/' + img_class + "_BASE1.png", semi_complete_merged1)
# cv2.imwrite(main_path + 'results/images/' + img_class + "_MERGED1.png", semi_complete_merged2)
# cv2.imshow("Complete", complete)
# cv2.imshow("Semi-Complete", semi_complete)

# (heatmap_merg_1, output_merg_1) = cam.overlay_heatmap(compl_heatmap[0], compl_heatmap[1], alpha=0.4)
# cv2.imshow("MERGED_1", output_merg_1)
# (heatmap_merg_2, output_merg_2) = cam.overlay_heatmap(compl_heatmap[2], compl_heatmap[3], alpha=0.4)
# (heatmap_merg_3, output_merg_3) = cam.overlay_heatmap(output_merg_1, output_merg_2, alpha=0.4)
# output_merg_1 = merg_average_pic(compl_heatmap[0], compl_heatmap[1], compl_heatmap[0].shape)
# output_merg_2 = merg_average_pic(compl_heatmap[2], compl_heatmap[3], compl_heatmap[2].shape)
# output_merg_3 = merg_average_pic(output_merg_1, output_merg_2, compl_heatmap[0].shape)
# cv2.imshow("MERGED_2", output_merg_2)
# semi_complete_merged1 = np.hstack([compl_heatmap[0], compl_heatmap[1], compl_heatmap[2], compl_heatmap[3]])
# semi_complete_merged1 = imutils.resize(semi_complete_merged1, height=450)
# cv2.imshow("BASELINE", semi_complete_merged1)
# semi_complete_merged2 = np.hstack([output_merg_1, output_merg_2, output_merg_3])
# semi_complete_merged2 = imutils.resize(semi_complete_merged2, height=400)
# cv2.imshow("MERGED", semi_complete_merged2)
# cv2.imwrite(main_path + 'results/images/' + img_class + "_BASE2.png", semi_complete_merged1)
# cv2.imwrite(main_path + 'results/images/' + img_class + "_MERGED2.png", semi_complete_merged2)
# cv2.imshow("Complete", complete)
# cv2.imshow("Semi-Complete", semi_complete)

# cv2.waitKey(0)
# exit()
