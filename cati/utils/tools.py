import os
from shutil import copy, move

from cati.utils.config import *


def create_folder(folder):
    try:
        os.makedirs(folder)
    except OSError:
        print("Creation of the directory %s failed" % folder)
    else:
        print("Successfully created the directory %s " % folder)


def find_smali(path, paths):
    if path.endswith(".smali"):
        if "R$" not in path and "BuildConfig" not in path:
            paths.append(path)
    elif os.path.isdir(path):
        for subdirectories in os.listdir(path):
            find_smali(f'{path}/{subdirectories}', paths)


def save_txt(path, content, utf):
    if utf:
        fw = open(path, "w", encoding='utf-8')
    else:
        fw = open(path, "w")
    fw.write(content)
    fw.close()


def create_dataset(apk_families, save_space, percentual):
    create_folder(f"{DATASETS}/dataset_{timeExec}")
    dataset_path = f"{DATASETS}/{timeExec}"
    create_folder(f"{dataset_path}/training")
    create_folder(f"{dataset_path}/training/train")
    create_folder(f"{dataset_path}/training/val")
    create_folder(f"{dataset_path}/test")
    for family in apk_families:
        create_folder(f"{dataset_path}/training/val/{family}")
        create_folder(f"{dataset_path}/training/train/{family}")
        create_folder(f"{dataset_path}/test/{family}")
        training = apk_families[family] * percentual/100
        validation = training * 0.20
        training -= validation
        # print(f"{family} \nTraining: {training} \nvalidation: {validation} \nTotale {apk_families[family]}")
        i = 0
        if save_space:
            for file in os.listdir(f"{DECOMPILED}/{family}"):
                if i < validation:
                    move(f"{RESULTS}/{family}/{file}.png", f"{dataset_path}/training/val/{family}")
                elif i < training:
                    move(f"{RESULTS}/{family}/{file}.png", f"{dataset_path}/training/train/{family}")
                else:
                    move(f"{RESULTS}/{family}/{file}.png", f"{dataset_path}/training/test/{family}")
                i += 1
        else:
            for file in os.listdir(f"{DECOMPILED}/{family}"):
                if i < validation:
                    copy(f"{RESULTS}/{family}/{file}.png", f"{dataset_path}/training/val/{family}")
                elif i < training:
                    copy(f"{RESULTS}/{family}/{file}.png", f"{dataset_path}/training/train/{family}")
                else:
                    copy(f"{RESULTS}/{family}/{file}.png", f"{dataset_path}/test/{family}")
                i += 1

