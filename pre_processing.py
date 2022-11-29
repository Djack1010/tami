import argparse
import datetime
import os

from utils import config
import time
from utils.generic_utils import print_log
from utils.convert_binary2image import binary2image


def parse_args():
    parser = argparse.ArgumentParser(prog="python pre_processing.py",
                                     description='Tool for Analyzing Malware represented as Images')
    group = parser.add_argument_group('Arguments')
    # OPTIONAL Arguments
    group.add_argument('-d', '--dataset', required=True, type=str, default=None,
                       help='the dataset path, must have the folder structure: training/train, training/val and test,'
                            'in each of this folders, one folder per class (see dataset_test)')
    group.add_argument('--mode', required=False, type=str, default='rgb-gray', choices=['rgb-gray', 'rgb', 'gray'],
                       help="Choose which mode run between 'rgb-gray' (default), 'rgb', and 'gray'."
                            "The 'rgb-gray' will convert the dataset in both grayscale and rgb colours, while the "
                            "other two modes ('rgb' and 'gray') only in rgb colours and grayscale, respectively.")
    group.add_argument('-v', '--version', action='version', version=f'{parser.prog} version {config.__version__}')
    arguments = parser.parse_args()
    return arguments


def _check_args(arguments):
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
    return


if __name__ == '__main__':
    # SET main_path at runtime with the absolute path of the project root folder
    path_list = os.path.realpath(__file__).split("/")[:-1]
    config.main_path = '/'.join(path_list) + '/'

    # START time and parse/check arguments
    start = time.perf_counter()
    args = parse_args()
    _check_args(args)

    # GLOBAL SETTINGS FOR THE EXECUTIONS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print_log("STARTING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)

    # Print information on log
    # EXECUTION Info
    print_log(f"INFO EXECUTION:"
              f"\nmode = {args.mode}"
              f"\ndataset = {args.dataset}"
              f"\n----------------")

    binary2image(config.main_path + args.dataset, width=None, thread_number=5, mode=args.mode)

    # Check if any 'ERROR!' print by the thread in the log file
    # TODO: naive approach, improve checks and errors handling
    errors_on_logfile = False
    with open(config.main_path + 'results/exec_logs/' + config.timeExec + ".results", 'r') as logfile:
        errors_on_logfile = True if 'ERROR!' in logfile.read() else False

    if errors_on_logfile:
        print_log(f"WARNING! There are some 'ERROR!'(s) in the logfile "
                  f"'results/exec_logs/{config.timeExec}.results'! You should read carefully the logfile!",
                  print_on_screen=True)

    print_log("ENDING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)

    end = time.perf_counter()
    print()
    print_log("EX. TIME: {} ".format(str(datetime.timedelta(seconds=end - start))), print_on_screen=True)
