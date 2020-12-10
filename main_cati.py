import argparse
import os
import re
import time

from utils.config import timeExec
import cati.utils.opcode as opcode
import cati.utils.tools as tools
import cati.utils.image as image
import cati.utils.process_data as process_data
from cati.utils.cati_config import DECOMPILED, RESULTS


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
    group.add_argument('-o', '--output_name', required=False, type=str, default="date",
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


if __name__ == "__main__":
    '''Converts the classes in the decompiled directories,
    saving data of the class itself and converting them in image'''
    start = time.perf_counter()

    args = parse_args()
    _check_args(args)

    apk = {}
    converter = opcode.Converter()

    if not os.path.isdir(f"{RESULTS}/{args.output_name}_{timeExec}"):
        tools.create_folder(f"{RESULTS}/{args.output_name}_{timeExec}")
    RESULTS = f"{RESULTS}/{args.output_name}_{timeExec}"

    for family in os.listdir(DECOMPILED):
        if os.path.isdir(f'{DECOMPILED}/{family}'):
            apk[family] = 0
            if not os.path.isdir(f"{RESULTS}/{family}"):
                tools.create_folder(f"{RESULTS}/{family}")
            for file in os.listdir(f'{DECOMPILED}/{family}'):
                apk[family] += 1

                if args.results:
                    smali_paths = []
                    smali_folder = f"{DECOMPILED}/{family}/{file}"
                    for subdirectory in os.listdir(smali_folder):
                        if "assets" not in subdirectory and "original" not in subdirectory and "res" not in subdirectory:
                            tools.find_smali(f"{smali_folder}/{subdirectory}", smali_paths)

                    general_content = ""
                    class_legend = ""
                    smali_k = {}
                    close = 0
                    for smali in smali_paths:
                        class_name = smali.replace(f'{DECOMPILED}/{family}', "")

                        fr = open(smali, "r")
                        encoded_content = converter.encoder(fr.read())
                        fr.close()

                        # calculating number of lines and characters of the encoded class
                        num_line = encoded_content.count('\n')
                        num_character = len(encoded_content)

                        # saving number of character and coordinates of begin and finish of the class
                        smali_k[class_name] = num_character
                        begin = close + 1
                        close = begin + num_line
                        class_legend += f"{class_name} [{begin},{close}]\n"
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

                    # saving the .txt containing all the compressed classes
                    tools.save_txt(f"{RESULTS}/{family}/{file}.txt", general_content, True)

                    # saving the legend of the classes
                    tools.save_txt(f"{RESULTS}/{family}/{file}_legend.txt", class_legend, False)

                    # saving the legend of the classes in the image
                    tools.save_txt(f"{RESULTS}/{family}/{file}_PNG_Legend.txt", image.legend_of_image(dim, smali_k), True)

    if args.storage:
        process_data.create_dataset(apk, args.output_name, args.image_size,
                                    args.training, args.validation)
