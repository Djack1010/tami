import argparse

from cati.utils.opcode import *
from cati.utils.tools import *
from cati.utils.process_data import *

apk = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description='APK converter in OPCode and than in PNG gray scaled image')
    group = parser.add_argument_group('Arguments')
    # OPTIONAL Arguments
    group.add_argument('-s', '--storage', required=False, type=str, default='preprocess',
                       help='Save a copy of the PNG files in a dataset tree struct '
                            '[none (n) to do not create the dataset]')
    group.add_argument('-p', '--percentual', required=False, type=int, default=80,
                       help='Percentual of division between training and test, insert a number between 1 to 100'
                            ' to define how large training should be')
    # As default the division training/validation is 80/20
    group.add_argument('-d', '--dims', required=False, type=int, default=250,
                       help='Dimension of the PNG genereted by the APK')
    arguments = parser.parse_args()
    return arguments


def _check_args(arguments):
    if arguments.storage != "preprocess" and arguments.storage != "normal" \
            or not arguments.storage.startswith("p") and not arguments.storage.startswith("n"):
        print('Invalid storage choice, exiting...')
        exit()
        if arguments.training_test < 0 or arguments.training_test > 100:
            print('Invalid suddivision, exiting...')
            exit()


def loop_per_decompiled():
    """Elaborates the classes in the decompiled directories,
    saving data of the class itself and converting it in an image"""
    apk = {}
    for family in os.listdir(DECOMPILED):
        if os.path.isdir(f'{DECOMPILED}/{family}'):
            apk[family] = 0
            os.chdir(RESULTS)
            if not os.path.isdir(f"{RESULTS}/{family}"):
                create_folder(family)
            for file in os.listdir(f'{DECOMPILED}/{family}'):
                apk[family] += 1

                smali_paths = []  # Initialise the list
                smali_folder = f"{DECOMPILED}/{family}/{file}"
                for subdirectory in os.listdir(smali_folder):
                    if "assets" not in subdirectory and "original" not in subdirectory and "res" not in subdirectory:
                        find_smali(f"{smali_folder}/{subdirectory}", smali_paths)

                general_content = ""
                class_legend = ""
                smali_k = {}
                cease = 0
                for smali in smali_paths:
                    class_name = smali.replace(f'{DECOMPILED}/family', "")

                    fr = open(smali, "r")
                    encoded_content = converter.encoder(fr.read())
                    fr.close()

                    # calculating number of lines and characters of the encoded class
                    num_line = encoded_content.count('\n')
                    num_character = len(encoded_content)

                    # saving number of character and coordinates of begin and finish of the class
                    smali_k[class_name] = num_character
                    begin = cease + 1
                    cease = begin + num_line
                    class_legend += f"{class_name} starts at the {begin}th line and ends at {cease}th line\n"
                    general_content += encoded_content

                # creating the image on the whole converted text
                img, pix_map, dim, num_character, lines = img_generator(general_content)
                char_reader(general_content, pix_map, dim)

                img.save(f"{family}/{file}.png")

                img, pix_map, dim = rgb_image_generator(len(general_content))
                pixel_generator(smali_k, pix_map, dim)
                img.save(f"{family}/{file}_legend.png")

                # saving the .txt containing all the compressed classes
                save_txt(f"{family}/{file}.txt", general_content, True)

                # saving the legend of the classes
                save_txt(f"{family}/{file}_legend.txt", class_legend, False)

                # saving the legend of the classes in the image
                save_txt(f"{family}/{file}_PNG_Legend.txt", legend_of_image(dim, smali_k), True)
    return apk


if __name__ == "__main__":
    """MAIN"""
    start = time.perf_counter()

    args = parse_args()
    _check_args(args)

    converter = Converter()

    apk = loop_per_decompiled()

    if not args.storage.startswith('n'):
        create_dataset(apk, args.dims, args.percentual)
