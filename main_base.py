import argparse
import datetime
import re
import os
import numpy as np
import pathlib
import tensorflow as tf
from models_base.basic import BASIC
from models_base.VGG16 import VGG16_19
from utils.config import *
from utils.generic_utils import print_log
from utils.analyzing_data import multiclass_analysis
from utils.preprocessing_data import get_info_dataset


def main(arguments, class_info):

    print_log("STARTING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)
    print("LOADING AND PRE-PROCESSING DATA")

    dataset_base = main_path + arguments.dataset

    # Create dataset of filepaths
    train_paths_ds = tf.data.Dataset.list_files(dataset_base + "/training/train/*/*")
    val_paths_ds = tf.data.Dataset.list_files(dataset_base + "/training/val/*/*")
    final_training_paths_ds = tf.data.Dataset.list_files(dataset_base + "/training/*/*/*")
    test_paths_ds = tf.data.Dataset.list_files(dataset_base + "/test/*/*")

    # STATS
    size_train = class_info['train_size']
    size_val = class_info['val_size']
    size_test = class_info['test_size']
    class_names = class_info['class_names']
    nclasses = class_info['n_classes']

    # SELECTING MODELS
    model = _model_selection(arguments, nclasses)

    # Print information on log
    # EXECUTION Info
    print_log("INFO EXECUTION:"
              "\nmodel = {}\ndataset = {}"
              "\noutput_model = {}\nepochs = {}\nbatch_size = {}\ncaching = {}"
              "\n----------------"
              .format(arguments.model, arguments.dataset,
                      arguments.output_model, arguments.epochs, arguments.batch_size, arguments.caching))

    # DATA Info
    print_log("INFO DATA:"
              "\num_classes = {}\nclass_names= {}\nnsize_img= {}x{}\nSize train-val-test= {}-{}-{}"
              .format(nclasses, class_names, arguments.image_size, arguments.channels, size_train, size_val, size_test))
    for ds_class in class_names:
        print_log("{} : {}-{}-{} -> {}".format(ds_class, class_info['info'][ds_class]['TRAIN'],
                                               class_info['info'][ds_class]['VAL'],
                                               class_info['info'][ds_class]['TEST'],
                                               class_info['info'][ds_class]['TOT']))
    print_log("----------------")

    # --------------  TRAINING and VALIDATION part  --------------------

    #  Use Dataset.map to create a dataset of image, label pairs
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    lab_train_ds = train_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    lab_val_ds = val_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # Caching dataset in memory for big dataset (IF arguments.caching is set)
    train_ds, val_ds = prepare_ds(arguments.caching, lab_train_ds, lab_val_ds, "train", "val", arguments.batch_size)

    print_log('Start Training for {} epochs  '.format(arguments.epochs), print_on_screen=True)

    # Initialize callbacks for Tensorboard
    log_fit = "tensorboard_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback_fit = tf.keras.callbacks.TensorBoard(log_dir=log_fit, histogram_freq=1)

    train_results = model.fit(x=train_ds, batch_size=arguments.batch_size, epochs=arguments.epochs,
                              validation_data=val_ds, callbacks=[tensorboard_callback_fit])

    print_log("\ttraining loss: {} \n\ttraining acc:{} \n\tvalidation loss:{} \n\tvalidation acc:{}"
              .format(train_results.history['loss'], train_results.history['acc'],
                      train_results.history['val_loss'], train_results.history['val_acc']))

    del train_ds, val_ds

    # --------------  FINAL TRAINING and TEST part  --------------------

    #  Use Dataset.map to create a dataset of image, label pairs
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    lab_final_train_ds = final_training_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    lab_test_ds = test_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # NB The batch_val_test is set to 1 to make easier the calculation of the performance results
    fin_train_ds, test_ds = prepare_ds(arguments.caching, lab_final_train_ds, lab_test_ds, "fin_tr", "test",
                                       arguments.batch_size, batch_val_test=1)

    # Train the model over the entire total_training set and then test
    print_log('Start Final Training for {} epochs  '.format(arguments.epochs), print_on_screen=True)
    start_training = time.perf_counter()
    final_train_results = model.fit(x=fin_train_ds, batch_size=arguments.batch_size, epochs=arguments.epochs)
    end_training = time.perf_counter()
    print_log("\ttraining loss: {} \n\ttraining acc:{}" .format(final_train_results.history['loss'],
                                                                final_train_results.history['acc']))
    print_log("FINAL TRAINING TIME: {} ".format(str(datetime.timedelta(seconds=end_training - start_training))))

    # Test the trained model over the test set
    print_log('Start Test', print_on_screen=True)
    results = model.evaluate(test_ds)
    print_log("\ttest loss: {} \n\ttest accuracy: {}".format(results[0], results[1]), print_on_screen=True)
    print_log("\tPrec: {} \n\tRecall: {}".format(results[2], results[3]), print_on_screen=True)
    # F-measure calculated as (2 * Prec * Recall)/(Prec + Recall)
    print_log("\tF-Measure: {} \n\tAUC: {}".format((2*results[2]*results[3])/(results[2]+results[3]), results[4]),
              print_on_screen=True)

    cm, results_classes, to_print = multiclass_analysis(model, test_ds, class_names,
                                                        save_fig=main_path + "results/figures/CM_{}".format(timeExec))
    print_log("Results per classes", print_on_screen=True)
    print_log(to_print, print_on_screen=True)

    del fin_train_ds, test_ds

    # save model and architecture to single file
    if arguments.output_model is not None:
        model.save(main_path + 'model_saved/{}_m{}'.format(arguments.output_model, arguments.model))
        model.save_weights(main_path + 'model_saved/{}_m{}_weights'.format(arguments.output_model, arguments.model))

    print_log("ENDING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Deep Learning Image Malware Classification')
    group = parser.add_argument_group('Arguments')
    # REQUIRED Arguments
    group.add_argument('-m', '--model', required=True, type=str,
                       help='BASIC or VGG16')
    group.add_argument('-d', '--dataset', required=True, type=str,
                       help='the dataset to be used')
    # OPTIONAL Arguments
    group.add_argument('-o', '--output_model', required=False, type=str, default=None,
                       help='Name of model to output and store')
    group.add_argument('-e', '--epochs', required=False, type=int, default=10,
                       help='number of epochs')
    group.add_argument('-b', '--batch_size', required=False, type=int, default=32)
    group.add_argument('-i', '--image_size', required=False, type=str, default="250x1",
                       help='FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and channel of the images in input '
                            '(reshape will be applied)')
    group.add_argument('-w', '--weights', required=False, type=str, default=None,
                       help="If you do not want random initialization of the model weights "
                            "(ex. 'imagenet' or path to weights to be loaded)")
    # FLAGS
    group.add_argument('--exclude_top', dest='include_top', action='store_false',
                       help='Exclute the fully-connected layer at the top pf the network (default INCLUDE)')
    group.set_defaults(include_top=True)
    group.add_argument('--caching', dest='caching', action='store_true',
                       help='Caching dataset on file and loading per batches (IF db too big for memory)')
    group.set_defaults(caching=False)
    arguments = parser.parse_args()
    return arguments


def _check_args(arguments):
    if arguments.model != "BASIC" \
            and arguments.model != "VGG16":
        print('Invalid model choice, exiting...')
        exit()
    if re.match(r"^\d{2,4}x([13])$", arguments.image_size):
        img_size = arguments.image_size.split("x")[0]
        channels = arguments.image_size.split("x")[1]
        setattr(arguments, "image_size", int(img_size))
        setattr(arguments, "channels", int(channels))
    else:
        print('Invalid image_size, exiting...')
        exit()
    if not os.path.isdir(main_path + arguments.dataset):
        print('Cannot find dataset in {}, exiting...'.format(arguments.dataset))
        exit()
    # Check Dataset struct: should be in folder tree training/[train|val] e test
    if not os.path.isdir(main_path + arguments.dataset + "/test") or \
            not os.path.isdir(main_path + arguments.dataset + "/training/val") or \
            not os.path.isdir(main_path + arguments.dataset + "/training/train"):
        print("Dataset '{}' should contain folders 'test, training/train and training/val'...".format(
            arguments.dataset))
        exit()


def _model_selection(arguments, nclasses):
    model = None
    print("INITIALIZING MODEL")

    if arguments.model == "BASIC":
        model_class = BASIC(nclasses, arguments.image_size, arguments.channels)
    elif arguments.model == "VGG16":
        # NB. Setting include_top=True and thus accepting the entire struct, the input Shape MUST be 224x224x3
        # and in any case, channels has to be 3
        if arguments.channels != 3:
            print("VGG requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, exiting...")
            exit()
        model_class = VGG16_19(nclasses, arguments.image_size, arguments.channels,
                      weights=arguments.weights, include_top=arguments.include_top)
    else:
        print("model {} not implemented yet...".format(arguments.model))
        exit()

    model = model_class.build()
    return model


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


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds, batch_size, cache=True, shuffle_buffer_size=1000, loop=False):
    """
    cache:  If isinstance(cache, str), then represents the name of a
            directory on the filesystem to use for caching elements in this Dataset.
            Otherwise, the dataset will be cached in memory.
    """
    # IF it is a small dataset, only load it once and keep it in memory.
    # OTHERWISE use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    if loop:
        ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def prepare_ds(caching, train, val_test, cache_train, cache_val_test, batch_train, batch_val_test=None):
    if batch_val_test is None:
        batch_val_test = batch_train
    if caching:
        # delete previous cache files and store for this execution
        caching_file_base = main_path + "temp/"
        for f in os.listdir(caching_file_base):
            if "{}.tfcache".format(cache_train) in f or "{}.tfcache".format(cache_val_test) in f:
                os.remove(caching_file_base + f)
        train_ds = prepare_for_training(train, batch_size=batch_train,
                                        cache=caching_file_base + "{}.tfcache".format(cache_train))
        test_val_ds = prepare_for_training(val_test, batch_size=batch_val_test,
                                           cache=caching_file_base + "{}.tfcache".format(cache_val_test))

    else:
        train_ds = prepare_for_training(train, batch_size=batch_train)
        test_val_ds = prepare_for_training(val_test, batch_size=batch_val_test)

    return train_ds, test_val_ds


if __name__ == '__main__':
    start = time.perf_counter()
    args = parse_args()
    _check_args(args)

    # Check info of the dataset
    # STRUCT of class_info = {'class_names': np.array(string), 'n_classes': int,
    # "train_size": int, "val_size": int, "test_size": int, 'info': dict}
    # for name in class_info['class_names'] the info dict contains = {'TRAIN': int, 'VAL': int, 'TEST': int, 'TOT': int}
    class_info = get_info_dataset(args.dataset)

    # GLOBAL SETTINGS FOR THE EXECUTIONS
    # Reduce verbosity for Tensorflow Warnings and set dtype for layers
    # tf.keras.backend.set_floatx('float64')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    CHANNELS = args.channels
    IMG_DIM = args.image_size
    CLASS_NAMES = class_info['class_names']
    BATCH_SIZE = args.batch_size

    # Check if tensorflow can access the GPU
    device_name = tf.test.gpu_device_name()
    if not device_name:
        print('GPU device not found...')
    else:
        print('Found GPU at: {}'.format(device_name))

    main(args, class_info)
    end = time.perf_counter()
    print()
    print_log("EX. TIME: {} ".format(str(datetime.timedelta(seconds=end-start))), print_on_screen=True)

