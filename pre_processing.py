import argparse
import datetime
import os
import re

from utils import config
import time
from utils.generic_utils import print_log
from utils.convert_binary2image import binary2image
from utils.preprocessing_data import split_dataset


def parse_args():
    parser = argparse.ArgumentParser(prog="python pre_processing.py",
                                     description='Tool for Analyzing Malware represented as Images')
    group = parser.add_argument_group('Arguments')
    # OPTIONAL Arguments
    group.add_argument('-d', '--dataset', required=True, type=str, default=None,
                       help='the dataset path')
    group.add_argument('--mode', required=False, type=str, default='rgb-gray',
                       choices=['rgb-gray', 'rgb', 'gray', 'ds'],
                       help="Choose which mode run between 'rgb-gray' (default), 'rgb', 'gray', and 'ds'."
                            "The 'rgb-gray' will convert the dataset in both grayscale and rgb colours, while the "
                            "other two modes ('rgb' and 'gray') only in rgb colours and grayscale, respectively.")
    group.add_argument('-p', '--percentage', required=False, type=str, default='80-10-10',
                       help='Percentage for training, validation, and test set when --mode=ds. '
                            'FORMAT ACCEPTED = X-Y-Z , which represent the training (X), validation (Y) and test (Z) '
                            'percentage, respectively. DEFAULT value is 80-10-10')
    group.add_argument('-v', '--version', action='version', version=f'{parser.prog} version {config.__version__}')
    arguments = parser.parse_args()
    return arguments


def _check_args(arguments):
    if not os.path.isdir(arguments.dataset):
        if os.path.isdir(config.main_path + arguments.dataset):
            setattr(arguments, "dataset", config.main_path + arguments.dataset)
        else:
            print('Cannot find dataset in {}, exiting...'.format(arguments.dataset))
            exit()
    # Check Dataset struct: should be in folder tree training/[train|val] e test
    if arguments.mode != 'ds':
        if not os.path.isdir(arguments.dataset + "/test") or \
                not os.path.isdir(arguments.dataset + "/training/val") or \
                not os.path.isdir(arguments.dataset + "/training/train"):
            print(f"Dataset '{arguments.dataset}' should contain folders 'test, training/train and training/val'...")
            exit()
    else:  # ds mode, check dataset percentage
        if re.match(r"^\d{1,2}-\d{1,2}-\d{1,2}$", arguments.percentage):
            train_perc = int(arguments.percentage.split("-")[0])
            val_perc = int(arguments.percentage.split("-")[1])
            test_perc = int(arguments.percentage.split("-")[2])
            if (train_perc + val_perc + test_perc) != 100:
                print(f"Percentage {train_perc}-{val_perc}-{test_perc} does not sum up to 100, exiting...")
                exit()
            setattr(arguments, "dataset_percentages", [train_perc, val_perc, test_perc])
        else:
            print("Invalid percentage '-p X-Y-Z' (see --help), exiting...")
            exit()


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

    if args.mode in ['rgb-gray', 'rgb', 'gray']:
        binary2image(args.dataset, width=None, thread_number=5, mode=args.mode)
    elif args.mode == 'ds':
        split_dataset(args.dataset, args.dataset_percentages)

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
