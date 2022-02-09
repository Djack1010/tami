import subprocess
from tqdm import tqdm
import threading
import argparse
import cati.utils.tools as tools
from time import sleep

from cati.utils.cati_config import *
from cati.utils.tools import *


def parse_args():
    parser = argparse.ArgumentParser(description='Decompile APK files to SMALI code')
    group = parser.add_argument_group('Arguments')
    # OPTIONAL Arguments
    group.add_argument('-n', '--input_path', required=False, type=str, default=None,
                       help='Input path to smali files')
    group.add_argument('-o', '--output_path', required=False, type=str, default=None,
                       help='Output path to store results')
    arguments = parser.parse_args()
    return arguments


def apktool_thread(directory, label, filename):
    subprocess.call([f'apktool d {directory}/{label}/{filename}'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    shell=True)


if __name__ == "__main__":
    args = parse_args()

    if args.input_path is not None:
        APK_DIR = args.input_path
        if not os.path.exists(APK_DIR):
            print(f"Input folder {APK_DIR} not found, exiting...")
            exit()
    if args.output_path is not None:
        DECOMPILED = args.output_path
        if not os.path.exists('/'.join(DECOMPILED.split('/')[:-1])):
            print(f"Output folder {DECOMPILED} not found, exiting...")
            exit()
        else:
            tools.create_folder(f"{DECOMPILED}", recreate=True)

    threads = []
    for family in os.listdir(APK_DIR):
        print(f"CLASS {family}")
        if os.path.isdir(f"{APK_DIR}/{family}"):
            os.chdir(DECOMPILED)
            if not os.path.isdir(f"{DECOMPILED}/{family}"):
                create_folder(family)
            os.chdir(f"{DECOMPILED}/{family}")
            for file in tqdm(os.listdir(f"{APK_DIR}/{family}")):
                if file[-3:] == "apk":
                    if os.path.exists(f"{DECOMPILED}/{family}/{file}"):
                        continue
                    else:
                        new_thread = threading.Thread(target=apktool_thread, args=[APK_DIR, family, file])
                        while threading.activeCount() > 4:
                            sleep(0.5)
                        new_thread.start()
                        threads.append(new_thread)
                        # print(f"Output to {os.getcwd()} for {APK_DIR}/{file}")
                        # subprocess.call([f'apktool d {APK_DIR}/{family}/{file}'],
                        #                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    for t in threads:
        if t.is_alive():
            t.join()
