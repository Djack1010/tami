import subprocess

from cati.utils.cati_config import *
from cati.utils.tools import *

if __name__ == "__main__":
    for family in os.listdir(APK_DIR):
        if os.path.isdir(f"{APK_DIR}/{family}"):
            os.chdir(DECOMPILED)
            if not os.path.isdir(f"{DECOMPILED}/{family}"):
                create_folder(family)
            os.chdir(f"{DECOMPILED}/{family}")
            for file in os.listdir(f"{APK_DIR}/{family}"):
                if file[-3:] == "apk":
                    print(os.getcwd())
                    print(f'{APK_DIR}/{file}')
                    subprocess.call([f'apktool d {APK_DIR}/{family}/{file}'],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
