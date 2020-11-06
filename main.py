import argparse
import datetime
import re
import os
import tensorflow as tf
from models_code.basic import BASIC
from models_code.nedo import NEDO
from models_code.VGG16 import VGG16_19
from utils.config import *
from utils.generic_utils import print_log
import utils.handle_modes as modes
from utils.preprocessing_data import get_info_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Deep Learning Image-based Malware Classification')
    group = parser.add_argument_group('Arguments')
    # REQUIRED Arguments
    group.add_argument('-m', '--model', required=True, type=str,
                       help='BASIC, NEDO or VGG16')
    group.add_argument('-d', '--dataset', required=True, type=str,
                       help='the dataset path, must have the folder structure: training/train, training/val and test,'
                            'in each of this folders, one folder per class (see dataset_test)')
    # OPTIONAL Arguments
    group.add_argument('-o', '--output_model', required=False, type=str, default=None,
                       help='Name of model to store')
    group.add_argument('-l', '--load_model', required=False, type=str, default=None,
                       help='Name of model to load')
    group.add_argument('-t', '--tuning', required=False, type=str, default=None,
                       help='Run Keras Tuner for tuning hyperparameters, chose: [hyperband, random, bayesian]')
    group.add_argument('-e', '--epochs', required=False, type=int, default=10,
                       help='number of epochs')
    group.add_argument('-b', '--batch_size', required=False, type=int, default=32)
    group.add_argument('-i', '--image_size', required=False, type=str, default="250x1",
                       help='FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and channel of the images in input '
                            '(reshape will be applied)')
    group.add_argument('-w', '--weights', required=False, type=str, default=None,
                       help="If you do not want random initialization of the model weights "
                            "(ex. 'imagenet' or path to weights to be loaded), not available for all models!")
    group.add_argument('--mode', required=False, type=str, default='training',
                       help="Choose which mode run between 'training' (default), 'test'. The 'training' mode will run"
                            "a phase of training+validation on the training and validation set, while the 'test' mode"
                            "will run a phase of training+test on the training+validation and test set.")
    # FLAGS
    group.add_argument('--exclude_top', dest='include_top', action='store_false',
                       help='Exclude the fully-connected layer at the top of the network (default INCLUDE)')
    group.set_defaults(include_top=True)
    group.add_argument('--caching', dest='caching', action='store_true',
                       help='Caching dataset on file and loading per batches (IF db too big for memory)')
    group.set_defaults(caching=False)
    arguments = parser.parse_args()
    return arguments


def _check_args(arguments):
    if arguments.model != "BASIC" and arguments.model != "NEDO" and arguments.model != "VGG16":
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
    if arguments.mode != "training" and arguments.mode != "test":
        print('Invalid mode choice, exiting...')
        exit()
    elif arguments.tuning is not None and arguments.tuning != 'hyperband' and arguments.tuning != 'random' \
            and arguments.tuning != 'bayesian':
        print('Invalid tuning choice, exiting...')
        exit()


def _model_selection(arguments, nclasses):
    print("INITIALIZING MODEL")
    mod_class = None
    if arguments.model == "BASIC":
        mod_class = BASIC(nclasses, arguments.image_size, arguments.channels)
    elif arguments.model == "NEDO":
        mod_class = NEDO(nclasses, arguments.image_size, arguments.channels)
    elif arguments.model == "VGG16":
        # NB. Setting include_top=True and thus accepting the entire struct, the input Shape MUST be 224x224x3
        # and in any case, channels has to be 3
        if arguments.channels != 3:
            print("VGG requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, exiting...")
            exit()
        mod_class = VGG16_19(nclasses, arguments.image_size, arguments.channels,
                               weights=arguments.weights, include_top=arguments.include_top)
    else:
        print("model {} not implemented yet...".format(arguments.model))
        exit()

    return mod_class


if __name__ == '__main__':
    start = time.perf_counter()
    args = parse_args()
    _check_args(args)

    # Check info of the dataset
    # STRUCT of class_info = {'class_names': np.array(string), 'n_classes': int,
    # "train_size": int, "val_size": int, "test_size": int, 'info': dict}
    # for name in class_info['class_names'] the info dict contains = {'TRAIN': int, 'VAL': int, 'TEST': int, 'TOT': int}
    class_info, ds_info = get_info_dataset(args.dataset)

    # GLOBAL SETTINGS FOR THE EXECUTIONS
    # Reduce verbosity for Tensorflow Warnings and set dtype for layers
    # tf.keras.backend.set_floatx('float64')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Check if tensorflow can access the GPU
    device_name = tf.test.gpu_device_name()
    if not device_name:
        print('GPU device not found...')
    else:
        print('Found GPU at: {}'.format(device_name))

    print_log("STARTING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)

    # SELECTING MODELS
    model_class = _model_selection(args, class_info['n_classes'])

    # Initialize variables and logs
    modes.initialization(args, class_info, ds_info)

    if args.tuning is not None:
        modes.tuning(args, model_class, ds_info)
    elif args.load_model is not None:
        model = modes.load_model(args)
        modes.test(args, model, class_info, ds_info)
    else:
        model = model_class.build()

        if args.mode == 'training':
            modes.train_val(args, model, ds_info)
        elif args.mode == 'test':
            modes.train_test(args, model, class_info, ds_info)

        modes.save_model(args, model)

    print_log("ENDING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)

    end = time.perf_counter()
    print()
    print_log("EX. TIME: {} ".format(str(datetime.timedelta(seconds=end - start))), print_on_screen=True)
