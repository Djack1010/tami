import argparse
import datetime
import re
import os
import tensorflow as tf

from code_models.custom_code_models.standard_CNN import StandardCNN as b_cnn
from code_models.custom_code_models.standard_MLP import StandardMLP as b_mlp
from code_models.custom_code_models.custom_CNN import CustomCNN as c_cnn

from utils import config
import time
from utils.generic_utils import print_log
import utils.handle_modes as modes
from utils.preprocessing_data import get_info_dataset
from utils.gradcam_back_code import apply_gradcam


def parse_args():
    parser = argparse.ArgumentParser(prog="python train_test.py", description='Tool for Analyzing Malware represented as Images')
    group = parser.add_argument_group('Arguments')
    # REQUIRED Arguments
    group.add_argument('-m', '--model', required=True, type=str, choices=['DATA', 'LE_NET', 'ALEX_NET', 'STANDARD_CNN',
                                                                          'STANDARD_MLP', 'CUSTOM_CNN', 'VGG16',
                                                                          'VGG19', 'Inception', 'ResNet50',
                                                                          'MobileNet', 'DenseNet', 'EfficientNet',
                                                                          'QCNN'],
                       help='Choose the model to use between the ones implemented')
    group.add_argument('-d', '--dataset', required=True, type=str,
                       help='the dataset path, must have the folder structure: training/train, training/val and test,'
                            'in each of this folders, one folder per class (see dataset_test)')
    # OPTIONAL Arguments
    group.add_argument('-o', '--output_model', required=False, type=str, default=None,
                       help='Name of model to store')
    group.add_argument('-l', '--load_model', required=False, type=str, default=None,
                       help='Name of model to load')
    group.add_argument('-t', '--tuning', required=False, type=str, default=None, choices=['hyperband', 'random',
                                                                                          'bayesian'],
                       help='Run Keras Tuner for tuning hyperparameters, options: [hyperband, random, bayesian]')
    group.add_argument('-e', '--epochs', required=False, type=int, default=10,
                       help='number of epochs')
    group.add_argument('-b', '--batch_size', required=False, type=int, default=16)
    group.add_argument('-i', '--image_size', required=False, type=str, default="100x1",
                       help='FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and channel of the images in input '
                            '(reshape will be applied)')
    group.add_argument('-w', '--weights', required=False, type=str, default='imagenet',
                       help="If you do not want random initialization of the model weights "
                            "(ex. 'imagenet' or path to weights to be loaded), not available for all models!")
    group.add_argument('-r', '--learning_rate', required=False, type=float, default=0.01,
                       help="Learning rate for training models")
    group.add_argument('--mode', required=False, type=str, default='train-val', choices=['train', 'train-val',
                                                                                         'train-test', 'test',
                                                                                         'gradcam-only'],
                       help="Choose which mode run between 'train-val' (default), 'train-test', 'train', 'test'."
                            "The 'train-val' mode will run a phase of training and validation on the training and "
                            "validation set, the 'train-test' mode will run a phase of training on the "
                            "training+validation sets and then test on the test set, the 'train' mode will run only a "
                            "phase of training on the training+validation sets, the 'test' mode will run only a "
                            "phase of test on the test set. The 'gradcam' has been moved to 'post_processing.py'")
    group.add_argument('-v', '--version', action='version', version=f'{parser.prog} version {config.__version__}')
    # FLAGS
    group.add_argument('--exclude_top', dest='include_top', action='store_false',
                       help='Exclude the fully-connected layer at the top of the network (default INCLUDE)')
    group.set_defaults(include_top=True)
    group.add_argument('--no-caching', dest='caching', action='store_false',
                       help='Caching dataset on file and loading per batches (IF db too big for memory)')
    group.set_defaults(caching=True)
    group.add_argument('--no-classes', dest='classAnalysis', action='store_false',
                       help='In case of mode including test, skip results for each class (only cumulative results)')
    group.set_defaults(classAnalysis=True)
    arguments = parser.parse_args()
    return arguments


def _check_args(arguments):
    if re.match(r"^\d{2,4}x([13])$", arguments.image_size):
        img_size = arguments.image_size.split("x")[0]
        channels = arguments.image_size.split("x")[1]
        setattr(arguments, "image_size", int(img_size))
        setattr(arguments, "channels", int(channels))
    else:
        print('Invalid image_size, exiting...')
        exit()
    if not os.path.isdir(config.main_path + arguments.dataset):
        print('Cannot find dataset in {}, exiting...'.format(config.main_path + arguments.dataset))
        exit()
    # Check Dataset struct: should be in folder tree training/[train|val] e test
    if not os.path.isdir(config.main_path + arguments.dataset + "/test") or \
            not os.path.isdir(config.main_path + arguments.dataset + "/training/val") or \
            not os.path.isdir(config.main_path + arguments.dataset + "/training/train"):
        print("Dataset '{}' should contain folders 'test, training/train and training/val'...".format(
            arguments.dataset))
        exit()
    if "gradcam" in arguments.mode:
        print("Gradcam processing has been moved to 'post_processing.py', exiting...")
        exit()
    if arguments.tuning is not None and arguments.tuning != 'hyperband' and arguments.tuning != 'random' \
            and arguments.tuning != 'bayesian':
        print('Invalid tuning choice, exiting...')
        exit()


def _model_selection(model_choice, nclasses):
    print("INITIALIZING MODEL")
    mod_class = None
    if model_choice == "STANDARD_CNN":
        mod_class = b_cnn(nclasses, config.IMG_DIM, config.CHANNELS, learning_rate=config.LEARNING_RATE)
    elif model_choice == "CUSTOM_CNN":
        mod_class = c_cnn(nclasses, config.IMG_DIM, config.CHANNELS, learning_rate=config.LEARNING_RATE)
    elif model_choice == "STANDARD_MLP":
        mod_class = b_mlp(nclasses, config.VECTOR_DIM, learning_rate=config.LEARNING_RATE)
    elif model_choice == "LE_NET":
        from code_models.sota_code_models.Le_Net_CNN import LeNet
        mod_class = LeNet(nclasses, config.IMG_DIM, config.CHANNELS, learning_rate=config.LEARNING_RATE)
    elif model_choice == "ALEX_NET":
        from code_models.sota_code_models.Alex_Net_CNN import AlexNet
        mod_class = AlexNet(nclasses, config.IMG_DIM, config.CHANNELS, learning_rate=config.LEARNING_RATE)
    elif model_choice == "QCNN":
        try:
            from code_models.sota_code_models.QCNN_QConv import QCNNqconv
            # Print a warning if IMG_DIM bigger than a threshold, QCNN requires small size images
            suggestion = 50
            if config.IMG_DIM > suggestion:
                print_log(f"QCNN requires pretty small images, image size {config.IMG_DIM} could raise problems, "
                          f"suggest to keep image size smaller than {suggestion} with "
                          f"'-i {suggestion}x{config.CHANNELS}'", print_on_screen=True)
            mod_class = QCNNqconv(nclasses, config.IMG_DIM, config.CHANNELS, learning_rate=config.LEARNING_RATE)
        except (ModuleNotFoundError, tf.errors.NotFoundError) as e:
            print_log(str(e), print_on_screen=True)
            print_log("ERROR! Unfortunately, there are libraries conflict between the ones listed in "
                      "the requirements...\nQUICK_FIX: Run experiments with QCNN on a virtualenv/container installing "
                      "the 'full_requirements.txt' file ONLY! \n"
                      "COMMANDS:(1) docker/manual_build.sh --quantum (2) docker/run_container.sh --quantum",
                      print_on_screen=True)
            exit()
    elif model_choice == "VGG16":
        from code_models.sota_code_models.VGG16 import VGG16
        # NB. Setting include_top=True and thus accepting the entire struct, the input Shape MUST be 224x224x3
        # and in any case, channels has to be 3
        if config.CHANNELS != 3:
            print("VGG requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, exiting...")
            exit()
        mod_class = VGG16(nclasses, config.IMG_DIM, config.CHANNELS, weights=config.WEIGHTS,
                          learning_rate=config.LEARNING_RATE)
    elif model_choice == "VGG19":
        from code_models.sota_code_models.VGG19 import VGG19
        # NB. Setting include_top=True and thus accepting the entire struct, the input Shape MUST be 224x224x3
        # and in any case, channels has to be 3
        if config.CHANNELS != 3:
            print("VGG requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, exiting...")
            exit()
        mod_class = VGG19(nclasses, config.IMG_DIM, config.CHANNELS, weights=config.WEIGHTS,
                          learning_rate=config.LEARNING_RATE)
    elif model_choice == "ResNet50":
        from code_models.sota_code_models.ResNet50 import ResNet
        if config.CHANNELS != 3:
            print("ResNet50 requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, exiting...")
            exit()
        mod_class = ResNet(nclasses, config.IMG_DIM, config.CHANNELS, weights=config.WEIGHTS,
                           learning_rate=config.LEARNING_RATE)
    elif model_choice == "Inception":
        from code_models.sota_code_models.InceptionV3 import Inception
        if config.CHANNELS != 3:
            print("INCEPTION requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, exiting...")
            exit()
        mod_class = Inception(nclasses, config.IMG_DIM, config.CHANNELS, weights=config.WEIGHTS,
                              learning_rate=config.LEARNING_RATE)
    elif model_choice == "MobileNet":
        from code_models.sota_code_models.MobileNet import MobNet
        if config.CHANNELS != 3:
            print("MobileNet requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, exiting...")
            exit()
        mod_class = MobNet(nclasses, config.IMG_DIM, config.CHANNELS, weights=config.WEIGHTS,
                           learning_rate=config.LEARNING_RATE)
    elif model_choice == "DenseNet":
        from code_models.sota_code_models.Dense121 import DenseNet
        if config.CHANNELS != 3:
            print("Dense121 requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, exiting...")
            exit()
        mod_class = DenseNet(nclasses, config.IMG_DIM, config.CHANNELS, weights=config.WEIGHTS,
                             learning_rate=config.LEARNING_RATE)
    elif model_choice == "EfficientNet":
        from code_models.sota_code_models.EfficientNet import EfficientNet
        if config.CHANNELS != 3:
            print("EfficientNet requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, "
                  "exiting...")
            exit()
        mod_class = EfficientNet(nclasses, config.IMG_DIM, config.CHANNELS, weights=config.WEIGHTS,
                                 learning_rate=config.LEARNING_RATE)
    else:
        print("model {} not implemented yet...".format(model_choice))
        exit()

    return mod_class


if __name__ == '__main__':
    # SET main_path at runtime with the absolute path of the project root folder
    path_list = os.path.realpath(__file__).split("/")[:-1]
    config.main_path = '/'.join(path_list) + '/'

    # START time and parse/check arguments
    start = time.perf_counter()
    args = parse_args()
    _check_args(args)

    # Check info of the dataset
    # STRUCT of class_info = {'class_names': np.array(string), 'n_classes': int,
    # "train_size": int, "val_size": int, "test_size": int, 'info': dict}
    # for name in class_info['class_names'] the info dict contains = {'TRAIN': int, 'VAL': int, 'TEST': int, 'TOT': int}
    class_info, ds_info = get_info_dataset(config.main_path + args.dataset,
                                           update=True if args.model == "DATA" else False)

    # if model set to 'DATA', only updates database info and exit
    if args.model == "DATA":
        print("Dataset info updated, exiting...")
        exit()

    # GLOBAL SETTINGS FOR THE EXECUTIONS
    # Reduce verbosity for Tensorflow Warnings and set dtype for layers
    # tf.keras.backend.set_floatx('float64')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # TODO: checks if we want to perform operation on CPU or GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to disable GPU
    # Check if tensorflow can access the GPU
    device_name = tf.test.gpu_device_name()
    if not device_name:
        print_log('LOADING TAMI: GPU device not found...')
    else:
        print_log('LOADING TAMI: Found GPU at: {}'.format(device_name))

    print_log("STARTING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)

    config.CHANNELS = args.channels
    config.IMG_DIM = args.image_size
    config.VECTOR_DIM = args.image_size * args.image_size * args.channels
    config.LEARNING_RATE = args.learning_rate
    config.WEIGHTS = args.weights

    # SELECTING MODELS
    model_class = _model_selection(args.model, class_info['n_classes'])
    model = None

    # Initialize variables and logs
    modes.initialization(args, class_info, ds_info, model_class)

    # Special modes
    # If tuning, the model to use has specific architecture define by build_tuning function in model classes
    if args.tuning is not None:
        modes.tuning(args, model_class, ds_info)

    # Standard modes of training, validation and test
    else:

        # Create model, either load from memory or create from model class
        if args.load_model is not None:
            model = modes.load_model(args, required_img=config.IMG_DIM, required_chan=config.CHANNELS,
                                     required_numClasses=class_info['n_classes'])
        else:
            try:
                model = model_class.build()
            except ValueError as e:
                print("ERROR: {}".format(e))
                print("NB. check if image size is big enough (usually, at least 25x1)")
                exit()

        try:
            # Modes which required a training phase
            if args.mode == 'train':
                modes.train_test(args, model, class_info, ds_info, conclude_wt_test=False)
            elif args.mode == 'train-val':
                modes.train_val(args, model, ds_info)
            elif args.mode == 'train-test':
                modes.train_test(args, model, class_info, ds_info)
            elif args.mode == 'test':
                modes.test(args, model, class_info, ds_info)
        except tf.errors.ResourceExhaustedError as e:
            print_log(f"ERROR: {str(e)}", print_on_screen=True)
            print_log(f"HINT1: You may try to reduce the batch size with '-b BATCH_SIZE' "
                      f"(current value: {args.batch_size}).", print_on_screen=True)
            print_log("HINT2: If failed during validation, you can run in train-test mode to avoid validation phase",
                      print_on_screen=True)

    print_log("ENDING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)

    end = time.perf_counter()
    print()
    print_log("EX. TIME: {} ".format(str(datetime.timedelta(seconds=end - start))), print_on_screen=True)
