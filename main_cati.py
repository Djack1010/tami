import subprocess
import argparse
from tqdm import tqdm
import time
import os
import re


from cati.utils.cati_config import DECOMPILED, RESULTS
import cati.utils.process_data as process_data
import cati.utils.opcode as opcode
import cati.utils.tools as tools
import cati.utils.image as image
from utils.config import timeExec


def parse_args():
    parser = argparse.ArgumentParser(
        description='APK converter in OPCode and than in PNG')
    group = parser.add_argument_group('Arguments')
    # OPTIONAL Arguments
    group.add_argument('-t', '--training', required=False, type=int, default=80,
                       help='Percentage of data to be saved in training, insert a number between 10 to 90')
    group.add_argument('-v', '--validation', required=False, type=int, default=20,
                       help='Percentage of data to be saved in validation, insert a number between 10 to 90')
    group.add_argument('-i', '--image_size', required=False, type=str, default="250x1",
                       help='FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and channel of the images in input '
                            'default is [250x1] (reshape will be applied)')
    group.add_argument('-o', '--output_name', required=False, type=str, default="data",
                       help='Enter the name by which you want to call the output')
    # FLAGS
    group.add_argument('--no_storage', dest='storage', action='store_false',
                       help='Do not create a dataset in tami/DATASETS folder')
    group.set_defaults(storage=True)
    group.add_argument('--no_results', dest='results', action='store_false',
                       help='Do nothing in results folder, so no creation of legends or images of the'
                            ' smali files in cati/results folder')
    group.set_defaults(results=True)
    arguments = parser.parse_args()
    return arguments


def _check_args(arguments):
    if re.match(r"^\d{2,4}x([13])$", arguments.image_size):
        img_size = arguments.image_size.split("x")[0]
        channels = arguments.image_size.split("x")[1]
        setattr(arguments, "image_size", int(img_size))
        setattr(arguments, "channels", int(channels))
    else:
        print('Invalid image_size, exiting...')
        exit()
    if not 10 <= arguments.training <= 90:
        print('Invalid training partition, exiting...')
        exit()
    if not 10 <= arguments.validation <= 90:
        print('Invalid validation partition, exiting...')
        exit()


def check_files(FOLDER):
    families = []
    for folder in os.listdir(DECOMPILED):
        if os.path.isdir(f'{DECOMPILED}/{folder}'):
            tools.create_folder(f"{FOLDER}/{folder}")
            families.append(folder)
    return families


if __name__ == "__main__":
    '''Converts the classes in the decompiled directories,
    saving data of the class itself and converting them in image'''
    start = time.perf_counter()

    args = parse_args()
    _check_args(args)

    RESULTS = f"{RESULTS}/{args.output_name}"

    apk = {}
    converter = opcode.Converter()

    tools.create_folder(f"{RESULTS}")

    FAMILIES = check_files(RESULTS)

    print('Processing of the macro folder')
    for family in tqdm(FAMILIES):
        apk[family] = 0
        for file in tqdm(os.listdir(f'{DECOMPILED}/{family}')):
            apk[family] += 1

            if args.results:
                smali_paths = []
                smali_folder = f"{DECOMPILED}/{family}/{file}"
                for subdirectory in os.listdir(smali_folder):
                    if "assets" not in subdirectory and "original" not in subdirectory and "res" not in \
                            subdirectory:
                        tools.find_smali(f"{smali_folder}/{subdirectory}", smali_paths)
                if not smali_paths:
                    print(f'In folder {family} the file {file} will be removed cause it is empty')
                    subprocess.call([f'rm -r {DECOMPILED}/{family}/{file}'],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
                else:
                    general_content = ""
                    smali_k = {}
                    close = 0
                    for smali in smali_paths:
                        class_name = smali.replace(f'{DECOMPILED}/{family}', "")

                        fr = open(smali, "r")
                        encoded_content = converter.encoder(fr.read())
                        fr.close()

                        # calculating number of characters of the encoded class
                        num_character = len(encoded_content)

                        # saving number of character
                        smali_k[class_name] = num_character
                        general_content += encoded_content

                    # creating the image on the whole converted text
                    if args.channels == 1:
                        img, pix_map, dim = image.img_generator(general_content, True)
                        image.char_reader_greyscale(general_content, pix_map, dim)
                    else:
                        img, pix_map, dim = image.img_generator(general_content, False)
                        image.char_reader_colorful(general_content, pix_map, dim)

                    img.save(f"{RESULTS}/{family}/{file}.png")

                    img, pix_map, dim = image.legend_image_generator(len(general_content))
                    image.legend_pixel_generator(smali_k, pix_map, dim)
                    img.save(f"{RESULTS}/{family}/{file}_legend.png")

                    # saving the legend of the classes in the image
                    tools.save_txt(f"{RESULTS}/{family}/{file}_PNG_Legend.txt", image.legend_of_image(dim,
                                                                                                      smali_k))

    if args.storage:
        process_data.create_dataset(apk, args.output_name, args.image_size,
                                    args.training, args.validation)
