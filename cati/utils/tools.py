import os
from shutil import copy, move

from PIL import Image

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

