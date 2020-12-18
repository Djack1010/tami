from datetime import datetime
from tqdm import tqdm
import subprocess
import argparse
import os
import re

from cati.utils.cati_config import DECOMPILED, RESULTS, APK_DIR
import cati.utils.process_data as process_data
import cati.utils.opcode as opcode
import cati.utils.tools as tools
import cati.utils.image as image


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
                       help='FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and Channel of the images in input '
                            'default is [250x1] (reshape will be applied)')
    group.add_argument('-o', '--output_name', required=False, type=str, default="data",
                       help='Enter the name by which you want to call the output')
    # FLAGS
    group.add_argument('--no_storage', dest='storage', action='store_false',
                       help='Do not create a dataset in tami/DATASETS')
    group.set_defaults(storage=True)
    group.add_argument('--no_results', dest='results', action='store_false',
                       help='Do nothing in results folder, so no creation of images or legends of the'
                            ' smali files in cati/RESULTS')
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


def _check_files(RES_FOLDER):
    families = []
    for folder in os.listdir(DECOMPILED):
        if os.path.isdir(f'{DECOMPILED}/{folder}'):
            tools.create_folder(f"{RES_FOLDER}/{folder}")
            families.append(folder)
    return families


if __name__ == "__main__":
    '''Converts the classes in the decompiled directories,
    saving data of the class itself and converting them in image'''
    start = datetime.now()
    print(f'Start time {start}')

    args = parse_args()
    _check_args(args)

    converter = opcode.Converter()

    RESULTS = f"{RESULTS}/{args.output_name}"
    tools.create_folder(f"{RESULTS}")
    FAMILIES = _check_files(RESULTS)

    apk = {}

    print('Initialization completed...')

    for family in FAMILIES:
        apk[family] = 0
        file_progressing = tqdm(os.listdir(f'{DECOMPILED}/{family}'),
                                position=0, unit=' file', bar_format='')
        for file in file_progressing:
            apk[family] += 1
            if args.results:
                name = file[0:10]
                file_progressing.bar_format = '{desc}{percentage:3.0f}%|{bar:20}{r_bar}'
                file_progressing.set_description(f'Processing the file ({name}...) of the folder ({family})')

                smali_paths = []
                smali_folder = f"{DECOMPILED}/{family}/{file}"
                # recursive function for the search of smali
                for subdirectory in os.listdir(smali_folder):
                    if "assets" not in subdirectory and "original" not in subdirectory and "res" not in \
                            subdirectory:
                        tools.find_smali(f"{smali_folder}/{subdirectory}", smali_paths)
                if not smali_paths:
                    # this command will remove unuseful apk
                    subprocess.call([f'rm -r {DECOMPILED}/{family}/{file} {APK_DIR}/{family}/{file}'],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
                else:
                    general_content = ""
                    smali_k = {}
                    end = 0
                    for smali in smali_paths:
                        class_name = smali.replace(f'{DECOMPILED}/{family}', "")
                        fr = open(smali, "r")
                        encoded_content = converter.encoder(fr.read())
                        fr.close()

                        # saving number of character and encoded content
                        num_character = len(encoded_content)
                        smali_k[class_name] = num_character
                        general_content += encoded_content

                    # creating the image on the whole converted text
                    if args.channels == 1:  # greyscale
                        img, pix_map, dim = image.img_generator(general_content, True)
                        image.char_reader_greyscale(general_content, pix_map, dim)
                    else:  # colorful
                        img, pix_map, dim = image.img_generator(general_content, False)
                        image.char_reader_colorful(general_content, pix_map, dim)
                    img.save(f"{RESULTS}/{family}/{file}.png")

                    # saving the png with the class division
                    img, pix_map, dim = image.legend_image_generator(len(general_content))
                    image.legend_pixel_generator(smali_k, pix_map, dim)
                    img.save(f"{RESULTS}/{family}/{file}_class.png")

                    # saving the legend of the classes in the image
                    tools.save_txt(f"{RESULTS}/{family}/{file}_legend.txt", image.legend_of_image(dim, smali_k))
    if args.result:
        print('Creation of the images completed')

    if args.storage:
        process_data.create_dataset(apk, args.output_name, args.image_size,
                                    args.training, args.validation)
        print('Creation of the dataset completed')
    print(f'Done\nTotal time: {datetime.now() - start}')
