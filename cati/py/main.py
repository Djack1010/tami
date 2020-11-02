import os
from utils.opcode import *
from utils.config import *
from utils.image import *
from utils.class_colored import *


def loop_per_decompiled(decompiled_files):
    """Elaborates the classes in the decompiled directories,
    saving data of the class itself and converting it in an image"""
    for file in decompiled_files:
        smali_path = \
            f"{DECOMPILED}/{file}/smali/com/example/{os.listdir(f'{DECOMPILED}/{file}/smali/com/example')[0]}"
        general_content = ""
        class_legend = ""
        smali_k = {}
        end = 0
        for smali in os.listdir(smali_path):
            if not smali.startswith("R$") and not smali.startswith("BuildConfig"):
                class_name = smali.split(".")
                fr = open(f"{smali_path}/{smali}", "r")
                encoded_content = converter.encoder(fr.read())
                fr.close()

                # calculating number of lines and characters of the encoded class
                num_line = encoded_content.count('\n')
                num_character = len(encoded_content)

                # saving number of character and coordinates of begin and finish of the class
                smali_k[class_name[0]] = num_character
                start = end + 1
                end = start + num_line
                class_legend += f"{class_name[0]} starts at the {start}th line and ends at {end}th line\n"
                general_content += encoded_content

        # creating the image on the whole converted text
        img, pix_map, dim, num_character, lines = img_generator(general_content)
        char_reader(general_content, pix_map, dim)

        img.save(f"{file}/{file}.png")

        img, pix_map, dim = rgb_image_generator(len(general_content))
        pixel_generator(smali_k, pix_map, dim)

        img.save(f"{file}/{file}Legend.png")

        # saving the .txt containing all the compressed classes
        fw = open(f"{file}/{file}.txt", "w", encoding='utf-8')
        fw.write(general_content)
        fw.close()

        # saving the legend of the classes
        fw = open(f"{file}/TXT Legend.txt", "w")
        fw.write(class_legend)
        fw.close()

        # saving the legend of the classes in the image
        fw = open(f"{file}/PNG Legend.txt", "w", encoding='utf-8')
        fw.write(legend_of_image(dim, smali_k))
        fw.close()


def checking_file():
    """Check if in the folder there are decompiled apk directories, if so load them in a list to be processed"""
    for file in os.listdir(DECOMPILED):
        # Seleziono solo le cartelle dei file scompattati
        if os.path.isdir(f"{DECOMPILED}/{file}"):
            # Controllo che non sia stata già creata una cartella apposita
            if not os.path.isdir(f"{RESULTS}/{file}"):
                try:
                    os.makedirs(file)
                except OSError:
                    print("Creation of the directory %s failed" % file)
                else:
                    print("Successfully created the directory %s " % file)
            yield file


if __name__ == "__main__":
    """MAIN"""
    os.chdir(RESULTS)
    converter = Converter()
    files = list(checking_file())
    loop_per_decompiled(files)
