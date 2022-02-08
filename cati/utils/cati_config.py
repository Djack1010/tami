from utils.config import main_path, timeExec
import os

# SET main_path at runtime with the absolute path of the project root folder
path_list = os.path.realpath(__file__).split("/")[:-3]
main_path = '/'.join(path_list) + '/'

APK_DIR = f"{main_path}cati/sample"
DECOMPILED = f"{main_path}cati/decompiled"
DICTIONARY = f"{main_path}cati/Dalvik_to_OPC.txt"
DICTIONARYrawbytes = f"{main_path}cati/Dalvik_to_OPC_RAWBYTES.txt"
RESULTS = f"{main_path}cati/RESULTS"
DATASETS = f"{main_path}DATASETS"
