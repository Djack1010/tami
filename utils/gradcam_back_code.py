import threading
from tqdm import tqdm
from time import sleep
from code_models.gradcams.first_gradcam import GradCAM as Gradcam_Standard
from code_models.gradcams.second_gradcam import GradCAM as Gradcam_Test
from code_models.gradcams.gradcam_utils import overlay_heatmap
from utils import config
from utils.handle_modes import process_path
import tensorflow as tf
import os
import numpy as np
import imutils
import random
import cv2


def from_pic_to_code(lock, heatmap, filename, cati, results_path, smali_data):
    if not os.path.exists(f"{cati}/{filename}.png"):
        print(f"ERROR, file {cati}/{filename}.png does not exist...")
        exit()
    else:
        cati_img = cv2.imread(f"{cati}/{filename}.png")
        # convert image to float, necessary before resizing it
        cleaned_heatmap = heatmap.reshape((heatmap.shape[0], heatmap.shape[1])) \
            .astype('float32')
        cleaned_heatmap = cv2.resize(cleaned_heatmap, (cati_img.shape[0], cati_img.shape[1]))
        relevant_pixels = []
        for y in range(cleaned_heatmap.shape[1]):
            for x in range(cleaned_heatmap.shape[0]):
                if cleaned_heatmap[x, y] > 0:
                    relevant_pixels.append(enumerate_pixel(x, y, cleaned_heatmap))

        mapped_smali = []
        with open(f"{cati}/{filename}_legend.txt", "r") as legend:
            for line in legend:
                strm = line.strip().replace("[", "").replace("]", "")
                smali_class = strm.split(" ")[0]
                # start = enumerate_pixel(int(strm.split(" ")[1].split(",")[0]),
                #                        int(strm.split(" ")[1].split(",")[1]),
                #                        cati_img, start_zero=False)
                end = enumerate_pixel(int(strm.split(" ")[2].split(",")[0]),
                                      int(strm.split(" ")[2].split(",")[1]),
                                      cati_img, start_zero=False)
                # print(f"{smali_class} {start} {end}")
                mapped_smali.append((int(end), smali_class))

        smali_to_look_into = []
        for rp in relevant_pixels:
            for ms in mapped_smali:
                if rp < ms[0]:

                    if ms[1] not in smali_to_look_into and "google/android" not in ms[1]:
                        smali_to_look_into.append(ms[1])

                        smali_name = ms[1].split("/")[-1]
                        with lock:
                            if smali_name not in smali_data:
                                smali_data[smali_name] = {'occ': 1, 'paths': [ms[1]]}
                            else:
                                smali_data[smali_name]['occ'] += 1
                                smali_data[smali_name]['paths'].append(ms[1])

                    break

        with open(results_path + f"/smali/{filename}.txt", "w") as to_analyse:
            for stli in smali_to_look_into:
                to_analyse.write(stli + "\n")


def clean_heatmap(pic, threshold):

    if threshold < 0 or threshold > 254:
        print(f"ERROR clean_heatmap: threashold {threshold} should be between 0 and 255, exiting...")
        exit()

    pic_new = np.zeros(shape=(pic.shape[0], pic.shape[1]), dtype=int)

    for x in range(pic.shape[0]):
        for y in range(pic.shape[1]):
            pix = pic[x, y]
            if pix > threshold:
                pic_new[x, y] = pix

    return pic_new


def enumerate_pixel(x, y, pic, start_zero=True):
    if start_zero:
        return y * pic.shape[0] + x
    else:
        return (y - 1) * pic.shape[0] + (x - 1)


def apply_gradcam(arguments, model, class_info, cati=True):

    # initialize the gradient class activation map
    cam = None
    if arguments.mode == 'gradcam-standard':
        cam = Gradcam_Standard(model, target_layer_min_shape=arguments.shape_gradcam)
    elif arguments.mode == 'gradcam-test':
        cam = Gradcam_Test(model, target_layer_min_shape=arguments.shape_gradcam)
    else:
        print(f"Gradcam required not found, exiting...")
        exit()

    if cati:
        # hardcoded path to cati folder with decompiled results
        # TODO: improve checks on path
        # ASSUMING that dataset path is DATASETS/name_date_timestamp/
        cati_db = arguments.dataset[9:].split("_")[0]
        cati_path_base = config.main_path + f"/cati/RESULTS/{cati_db}/"
        if not os.path.isdir(cati_path_base):
            print(f"ERROR! cati db with smali class info not found in {cati_path_base}, exiting...")
            exit()

    # create folder in /results/images for this execution
    images_path = f"{config.main_path}results/images/{config.timeExec}_{arguments.load_model}_{arguments.mode}"
    os.mkdir(images_path)

    index = 0
    for img_class in class_info["class_names"]:

        index += 1
        print(f"GradCAM '{arguments.mode}' on output class '{img_class}' - {index} out of {len(class_info['class_names'])}")

        # Adding also a '/' to ensure path correctness
        label_path = config.main_path + arguments.dataset + "/test/" + img_class
        if cati:
            cati_path = cati_path_base + img_class

            # Initialize data struct for analyze heatmaps and decompiled smali
            smali_code = {}
            lock = threading.Lock()
            threads = []

        # Get all file paths in 'label_path' for the class 'label'
        files = [i[2] for i in os.walk(label_path)]

        if arguments.sample_gradcam is None:
            num_samples = len(files[0])
        else:
            num_samples = arguments.sample_gradcam if len(files[0]) >= arguments.sample_gradcam else len(files[0])

        # Randomly extract 'num_sample' from the file paths, in files there is a [[files_paths1, filepath2,...]]
        imgs = random.sample(files[0], num_samples)

        # create folders in /results/images/<TIME_MODEL> for this class
        class_images_path = f"{images_path}/{img_class}"
        if not os.path.isdir(class_images_path):
            os.mkdir(class_images_path)
            os.mkdir(class_images_path + '/heatmap')
            os.mkdir(class_images_path + '/complete')
            os.mkdir(class_images_path + '/highlights_heatmap')
            os.mkdir(class_images_path + '/smali')

        for i in tqdm(range(num_samples)):
            complete_path = label_path + "/" + imgs[i]
            img_filename = imgs[i].split(".")[0]

            # REPLACE _ with -, which may cause some problem for hardcoded format search
            if "_" in img_filename:
                img_filename = img_filename.replace('_', '-')

            # load the original image from disk (in OpenCV format) and then
            # resize the image to its target dimensions
            orig = cv2.imread(complete_path)
            # resized = cv2.resize(orig, (arguments.image_size, arguments.image_size))

            image, _ = process_path(complete_path)
            image = tf.expand_dims(image, 0)

            # use the network to make predictions on the input image and find
            # the class label index with the largest corresponding probability
            preds = model.predict(image)
            i = np.argmax(preds[0])

            # decode the ImageNet predictions to obtain the human-readable label
            # decoded = imagenet_utils.decode_predictions(preds)
            # (imagenetID, label, prob) = decoded[0][0]
            # label = "{}: {:.2f}%".format(label, prob * 100)
            correctness = f"WRONG{class_info['class_names'][int(i)]}" if img_class != class_info["class_names"][int(i)]\
                else f"{img_class}"
            label = "{} - {:.1f}%".format(correctness, preds[0][i] * 100)
            #print("[INFO] {}".format(label))

            # build the heatmap
            heatmap = cam.compute_heatmap(image, class_index=i)

            # resize heatmap to size of origin file and copy to stored later
            # at this point the heatmap contains integer value scaled [0, 255]
            heatmap_origin_size = cv2.resize(heatmap.copy(), (orig.shape[1], orig.shape[0]))

            # resize the heatmap to the original input image dimensions and overlay heatmap on top of the image
            (heatmap, output) = overlay_heatmap(cv2.resize(heatmap, (orig.shape[1], orig.shape[0])), orig, alpha=0.5)

            # resize images
            orig = imutils.resize(orig, width=400)
            heatmap = imutils.resize(heatmap, width=400)
            output = imutils.resize(output, width=400)

            # create a black background to include text
            black = np.zeros((35, orig.shape[1], 3), np.uint8)
            black[:] = (0, 0, 0)

            # concatenate vertically to the image
            orig = cv2.vconcat((black, orig))
            heatmap = cv2.vconcat((black, heatmap))
            output = cv2.vconcat((black, output))

            # write some text over each image
            cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
            cv2.putText(heatmap, "Heatmap", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
            cv2.putText(output, "Overlay with Heatmap", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))

            # display the original image and resulting heatmap and output image
            complete = np.hstack([orig, heatmap, output])

            cv2.imwrite(class_images_path + '/complete/complete_' + img_filename + '.png', complete)

            # store heatmaps
            cv2.imwrite(class_images_path + f'/heatmap/heatmap{correctness if "WRONG" in correctness else ""}_' +
                        img_filename + '.png', heatmap_origin_size)

            # clean and store heatmaps
            cleaned_heatmap = clean_heatmap(heatmap_origin_size, 150)
            cv2.imwrite(class_images_path + f'/highlights_heatmap/heatmap{correctness if "WRONG" in correctness else ""}_'
                        + img_filename + '.png', cleaned_heatmap)

            if cati:
                # lock, heatmap, filename, cati, results_path, smali_data
                new_thread = threading.Thread(target=from_pic_to_code, args=[lock, cleaned_heatmap, img_filename, cati_path,
                                                                             class_images_path, smali_code])
                while threading.activeCount() > 6:
                    sleep(0.5)
                new_thread.start()
                threads.append(new_thread)

        if cati:
            for t in threads:
                if t.is_alive():
                    t.join()

            # convert dict to list -> the dict was faster for threading search/insert on shared struct, but now we prefer
            # list to order the element by occurencies
            smali_code_list = []
            for e in smali_code:
                smali_code_list.append([smali_code[e]['occ'], e, smali_code[e]['paths']])
            smali_code_list.sort(key=lambda x: x[0], reverse=True)

            with open(class_images_path + "/SMALI_CLASS.txt", "w") as to_analyse:
                for i in range(20):
                    to_analyse.write(f"{smali_code_list[i][1]} {smali_code_list[i][0]} {smali_code_list[i][2]}\n")
