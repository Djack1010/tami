import argparse
import datetime
import os
import tensorflow as tf
from utils.SSIM_metrics import IFIM_SSIM
from glob import glob

from utils import config
import time
from utils.generic_utils import print_log
import utils.handle_modes as modes
from utils.preprocessing_data import get_info_dataset
from utils.gradcam_back_code import apply_gradcam


def parse_args():
    parser = argparse.ArgumentParser(prog="python post_processing.py",
                                     description='Tool for Analyzing Malware represented as Images')
    group = parser.add_argument_group('Arguments')
    # OPTIONAL Arguments
    group.add_argument('-l', '--load_model', required=False, type=str, default=None, help='Name of model to load')
    group.add_argument('-d', '--dataset', required=False, type=str, default=None,
                       help='the dataset path, must have the folder structure: training/train, training/val and test,'
                            'in each of this folders, one folder per class (see dataset_test)')
    group.add_argument('-gl', '--sample_gradcam', required=False, type=int, default=None,
                       help="Limit gradcam to X samples randomly extracted from the test set")
    group.add_argument('-gs', '--shape_gradcam', required=False, type=int, default=1,
                       help="Select gradcam target layer with at least shapeXshape (for comparing different models)")
    group.add_argument('-sf', '--ssim_folders', required=False, nargs='*', default=None,
                       help="List of gradcam results folder to compare with IF-SSIM and IM-SSIM")
    group.add_argument('--mode', required=False, type=str, default='gradcam-standard', choices=['IFIM-SSIM',
                                                                                            'gradcam-standard',
                                                                                            'gradcam-test',
                                                                                            'gradcam-cati'],
                       help="Choose which mode run between 'gradcam-only' (default), 'gradcam-cati', 'IFIM-SSIM'"
                            "The 'gradcam-[cati|only]' will run the gradcam analysis on "
                            "the model provided. 'gradcam-only' will generate the heatmaps only, while 'gradcam-cati "
                            "will also run the cati tool to reverse process and select the code from the heatmap to "
                            "the decompiled smali (if provided, see cati README)")
    group.add_argument('-v', '--version', action='version', version=f'{parser.prog} version {config.__version__}')
    # FLAGS
    group.add_argument('--include_all', dest='include_all', action='store_true',
                       help='Include all possible heatmaps in the IFIM-SSIM analysis (default, choose a random subset)')
    group.set_defaults(include_all=False)
    arguments = parser.parse_args()
    return arguments


def _check_args(arguments):
    if arguments.dataset is not None:
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
    if "gradcam-" in arguments.mode:
        if arguments.load_model is None:
            print("ERROR! You need to load a model with '-l MODEL_NAME' for the gradcam analysis, exiting...")
            exit()
        if arguments.dataset is None:
            print("ERROR! You need to specify a dataset with '-d DATASET_PATH' for the gradcam analysis. "
                  "The test set will be used to generate the heatmaps. Exiting...")
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
    if "IFIM-SSIM" == arguments.mode and arguments.ssim_folders is None:
        print("ERROR! You need to specify which folders in 'results/images' should be compared with the IF-SSIM and "
              "the IM-SSIM. You need to provide a list of folder with the input '-sf FOLDER1 FOLDER2 ... FOLDERN'."
              " If you want to compare ALL the folders, provide input '-sf ALL'. exiting...")
        available_folders = [x.split('results/images/')[1] for x in glob(config.main_path + 'results/images/*/',
                                                                         recursive=True)]
        print(f"AVAILABLE FOLDERS: {' '.join(str(x) for x in available_folders)}")
        exit()


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
    if args.dataset is not None:
        class_info, ds_info = get_info_dataset(config.main_path + args.dataset)
    else:
        class_info, ds_info = None, None

    # GLOBAL SETTINGS FOR THE EXECUTIONS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to disable GPU
    # Check if tensorflow can access the GPU
    device_name = tf.test.gpu_device_name()
    if not device_name:
        print('GPU device not found...')
    else:
        print('Found GPU at: {}'.format(device_name))

    print_log("STARTING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)

    # Initialize variables and logs
    modes.initialization_postprocessing(args, class_info)

    if 'gradcam-' in args.mode:
        # LOADING MODEL
        model = modes.load_model(args, required_img=None, required_chan=None,
                                 required_numClasses=class_info['n_classes'])

        apply_gradcam(args, model, class_info, cati=True if args.mode == 'gradcam-cati' else False)

    elif args.mode == 'IFIM-SSIM':
        IFIM_SSIM(args)

    print_log("ENDING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)

    end = time.perf_counter()
    print()
    print_log("EX. TIME: {} ".format(str(datetime.timedelta(seconds=end - start))), print_on_screen=True)
