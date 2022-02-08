import os
import shutil


def create_folder(folder, recreate=False):
    if not os.path.isdir(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass
    elif recreate:
        shutil.rmtree(folder)
        try:
            os.makedirs(folder)
        except OSError:
            pass



def find_smali(path, paths):
    if path.endswith(".smali"):
        if "R$" not in path and "BuildConfig" not in path:
            paths.append(path)
    elif os.path.isdir(path):
        for subdirectories in os.listdir(path):
            find_smali(f'{path}/{subdirectories}', paths)
