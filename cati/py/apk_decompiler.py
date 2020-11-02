import os
import subprocess
from utils.config import *

if __name__ == "__main__":
    os.chdir(DECOMPILED)
    for file in os.listdir(APK_DIR):
        if file[-3:] == "apk":
            subprocess.call(['apktool', 'd', f'{APK_DIR}/{file}', '-f'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
