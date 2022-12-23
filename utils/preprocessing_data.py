import os
import numpy as np
import pickle
import pathlib
from random import shuffle, choice
from utils import config
import shutil
from utils.generic_utils import print_log


def get_info_dataset(dataset_path, update=False):
    # TODO: Implements some checks to verify edits to the dataset from last pickle.dump(data)
    storing_data_path = dataset_path + "/info.txt"

    if update and os.path.exists(dataset_path + "/info.txt"):
        os.remove(dataset_path + "/info.txt")

    if os.path.isfile(storing_data_path):
        with open(storing_data_path, 'rb') as filehandle:

            data = pickle.load(filehandle)
            class_info = data['class_info']
            ds_info = data['ds_info']

            # CHECKS if the paths stored match the DS
            # TODO: This check just pick 3 elements and check existence, can be improved
            if not os.path.exists(choice(ds_info['train_paths'])) or not os.path.exists(choice(ds_info['val_paths'])) \
                    or not os.path.exists(choice(ds_info['test_paths'])):
                print(f"Dataset paths seem incorrect, "
                      f"you should update the dataset info running 'python train_test.py -m DATA -d {dataset_path}")
                exit()
            # Shuffle elements
            else:
                shuffle(ds_info['train_paths'])
                shuffle(ds_info['val_paths'])
                shuffle(ds_info['final_training_paths'])
                shuffle(ds_info['test_paths'])

    else:

        # Create dataset filepaths
        train_paths = [os.path.join(r, file) for r, d, f in os.walk(dataset_path + "/training/train")
                       for file in f if ".png" in file or ".jpg" in file or ".jpeg" in file]
        val_paths = [os.path.join(r, file) for r, d, f in os.walk(dataset_path + "/training/val")
                     for file in f if ".png" in file or ".jpg" in file or ".jpeg" in file]
        final_training_paths = [os.path.join(r, file) for r, d, f in os.walk(dataset_path + "/training")
                                for file in f if ".png" in file or ".jpg" in file or ".jpeg" in file]
        test_paths = [os.path.join(r, file) for r, d, f in os.walk(dataset_path + "/test")
                      for file in f if ".png" in file or ".jpg" in file or ".jpeg" in file]

        ds_info = {'ds_type': 'images', 'train_paths': train_paths, 'val_paths': val_paths, 'test_paths': test_paths,
                   'final_training_paths': final_training_paths}

        temp_class_names = np.array([item.name for item in pathlib.Path(dataset_path + "/training/train").glob('*')])
        # Sort class_names to keep same order, which influence training in one-hot encore, over different machines
        class_names = np.sort(temp_class_names, axis=-1)
        nclasses = len(class_names)
        class_info = {"class_names": class_names, "n_classes": nclasses}

        # GENERAL STATS
        size_train = len(train_paths)
        size_val = len(val_paths)
        size_test = len(test_paths)

        class_info.update({"train_size": size_train, "val_size": size_val, "test_size": size_test, 'info': {}})

        for name in class_names:
            size_trainf = sum([len(files) for r, d, files in os.walk(dataset_path + "/training/train/{}".format(name))])
            size_valf = sum([len(files) for r, d, files in os.walk(dataset_path + "/training/val/{}".format(name))])
            size_testf = sum([len(files) for r, d, files in os.walk(dataset_path + "/test/{}".format(name))])
            class_info['info']["{}".format(name)] = {}
            class_info['info']["{}".format(name)]['TRAIN'] = size_trainf
            class_info['info']["{}".format(name)]['VAL'] = size_valf
            class_info['info']["{}".format(name)]['TEST'] = size_testf
            class_info['info']["{}".format(name)]['TOT'] = size_testf + size_valf + size_trainf

        with open(storing_data_path, 'wb') as filehandle:
            data = {'ds_info': ds_info, 'class_info': class_info}
            pickle.dump(data, filehandle)

    return class_info, ds_info


def split_dataset(dataset_path, percentages):
    # os.walk is a generator and calling next will get the first result in the form of a 3-tuple
    # (dirpath, dirnames, filenames).
    dataset = {}
    output_classes = next(os.walk(dataset_path))[1]
    warning = False

    for oc in output_classes:
        dataset[oc] = {"total": next(os.walk(f"{dataset_path}/{oc}"))[2]}
        # shuffle to randomize datset generation
        shuffle(dataset[oc]['total'])
        train_val = int(len(dataset[oc]['total']) * (percentages[0] / 100))
        val_val = int(len(dataset[oc]['total']) * (percentages[1] / 100))
        dataset[oc]['train'] = dataset[oc]['total'][:train_val]
        dataset[oc]['val'] = dataset[oc]['total'][train_val:train_val+val_val]
        dataset[oc]['test'] = dataset[oc]['total'][train_val+val_val:]

    # Create DS name based on the input dataset name
    dataset_name = dataset_path.split('/')[-1] if dataset_path.split('/')[-1] != "" else dataset_path.split('/')[-2]
    dataset_name = f"{dataset_name}_TAMIDS"

    print("---- PRE-DATASET GENERATION: recap on found samples ----")
    print(f"Output Classes ({len(output_classes)}): {output_classes}")
    print("Split dataset into 'OUTPUT_CLASS -> TRAINING - VALIDATION - TEST' samples")
    for oc in output_classes:
        print(f"{oc} -> {len(dataset[oc]['train'])} - {len(dataset[oc]['val'])} - {len(dataset[oc]['test'])}")
        if len(dataset[oc]['train']) < 100 or len(dataset[oc]['val']) < 30 or len(dataset[oc]['test']) < 30:
            warning = True
    print(f"(Over)Writing the dataset in '{config.main_path}DATASETS/{dataset_name}'")
    if warning:
        print(f"WARNING! The dataset numbers seem incorrect (too few samples)")

    proceed = str(input("Do you confirm these dataset? [Y/n] \n"))
    if proceed.lower() == "n" or proceed.lower() == "no":
        print_log("User stopped the dataset split, exiting...")
        exit()

    # Delete previous DS (if present)
    if os.path.exists(f"{config.main_path}/DATASETS/{dataset_name}"):
        shutil.rmtree(f"{config.main_path}/DATASETS/{dataset_name}")

    # Copy files in the DS
    for oc in output_classes:
        os.makedirs(f"{config.main_path}/DATASETS/{dataset_name}/training/train/{oc}", exist_ok=True)
        os.makedirs(f"{config.main_path}/DATASETS/{dataset_name}/training/val/{oc}", exist_ok=True)
        os.makedirs(f"{config.main_path}/DATASETS/{dataset_name}/test/{oc}", exist_ok=True)
        for set in ['train', 'val', 'test']:
            for x in dataset[oc][set]:
                shutil.copy(f"{dataset_path}/{oc}/{x}",
                            f"{config.main_path}/DATASETS/{dataset_name}/"
                            f"{'' if set == 'test' else 'training/'}{set}/{oc}/{x}")

    print_log(f"Dataset created in '{config.main_path}DATASETS/{dataset_name}'", print_on_screen=True)
