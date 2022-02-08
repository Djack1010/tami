import math
import random

from PIL import Image


def legend_image_generator(num):
    dim = int(math.sqrt(num)) + 1
    img = Image.new('RGB', (dim, dim), 'white')
    pix_map = img.load()
    return img, pix_map, dim


def legend_pixel_generator(smali_dim, pixMap, dim):
    x, y = 0, 0  # coordinates pixelmap
    for name in smali_dim:

        R = random.randint(0, 128)
        G = random.randint(0, 128)
        B = random.randint(0, 128)

        i = 0
        color = (R, G, B)
        while i < smali_dim[name]:
            if y < dim:
                x, y, pixMap = image_filler(pixMap, color, x, y, dim)
            else:
                print("error image's dimension")
                break
            i += 1

    return pixMap


def img_generator(content, greyscale):
    """This function counts the parameters, like number of character and side of the square, to generate the image"""
    num_of_characters = len(content)

    # calculate the side of the square and instantiate the new image
    dim = int(math.sqrt(num_of_characters)) + 1
    if greyscale:
        quality = 'L'
    else:
        quality = 'RGB'
    img = Image.new(quality, (dim, dim), 'white')
    pix_map = img.load()
    return img, pix_map, dim


def char_reader_greyscale(content, pix_map, dim):
    """Convert the characters in pixel and load them in the pixel map"""
    x, y = 0, 0  # pixel_map coordinates
    for char in content:
        single_pixel = char_to_grayscale(char)

        # checking to not exit from the side of the square to fill the pixel map
        if y < dim:
            x, y, pix_map = fil_image(pix_map, single_pixel, x, y, dim)
        else:
            print("error image's dimension")
            break

    return pix_map


def char_reader_colorful(content, pix_map, dim):
    """Convert the characters in pixel and load them in the pixel map"""
    x, y = 0, 0  # pixel_map coordinates
    string = ""
    single_pixel = 0
    for char in content:
        string += char
        if len(string) == 3:
            single_pixel = char_to_color(string)
            string = ""

        # checking to not exit from the side of the square to fill the pixel map
        if y < dim:
            x, y, pix_map = fil_image(pix_map, single_pixel, x, y, dim)
        else:
            print("error image's dimension")
            break

    return pix_map


def char_to_grayscale(char):
    """Converts the character in a number to be represented in a pixel"""
    bit = "".join("{:8b}".format(ord(char)))
    pixel = (int(bit, 2))
    return pixel


def char_to_color(string):
    pixel_list = []
    for i in string:
        bit = "".join("{:8b}".format(ord(i)))
        pixel_list.append(int(bit, 2))
    pixel = tuple(pixel_list)
    return pixel


def image_filler(pixmap, pixel, x, y, dim):
    if x < dim:
        pixmap[x, y] = pixel
        x += 1
    else:
        x = 0
        y += 1
        pixmap[x, y] = pixel
        x += 1

    return x, y, pixmap


def fil_image(pixmap, pixel, x, y, dim):
    """Load the pixel in a matrix of pixel"""
    if x < dim:
        pixmap[x, y] = pixel
        x += 1
    else:
        x = 0
        y += 1
        pixmap[x, y] = pixel
        x += 1

    return x, y, pixmap


def save_legend(smali_dim, output, square_side=None):
    """
    These lines read the number of character of a class and calculate where it begins and ends inside the image
      IF square_side is None, the character are printed NOT as an image, but as raw bytes in line.
    """
    with open(output, "w") as out_file:
        end = 0
        for name in smali_dim:

            if square_side is not None:
                start = end + 1
                end = start + smali_dim[name]
                xs = start % square_side + 1
                ys = start // square_side + 1

                xe = end % square_side + 1
                ye = end // square_side + 1
                out_file.write(f"\n{name} [{xs},{ys}] [{xe},{ye}]")

            else:
                start = end + 1
                end = smali_dim[name]['len'] + 1
                out_file.write(f"\n{name} [{start}-{end}] METHODS: {smali_dim[name]['meth']}")
