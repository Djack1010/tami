import math

from PIL import Image


def img_generator(content):
    """This function counts the parameters, like number of character and side of the square, to generate the image"""
    num_of_lines = content.count('\n')
    num_of_characters = len(content)

    # calculate the side of the square and instantiate the new image
    dim = int(math.sqrt(num_of_characters)) + 1
    img = Image.new('L', (dim, dim), 'white')
    pix_map = img.load()
    return img, pix_map, dim, num_of_characters, num_of_lines


def char_reader(content, pix_map, dim):
    """Call the functions to convert the characters in pixel and to load them in the pixel map"""
    x, y = 0, 0  # pixel_map coordinates
    for char in content:

        # checking to not exit from the side of the square and call of the method to fill the pixel map
        single_pixel = char_to_grayscale(char)
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


def legend_of_image(square_side, smali_dim):
    """These lines read the number of character of the class and calculate where it begins and ends inside the image"""
    image_legend = ""
    first = True
    end = 0
    for name in smali_dim:
        if first:
            end = smali_dim[name] + 1
            xe = end % square_side + 1
            ye = end // square_side + 1
            image_legend = f"{name} starts: 1x and 1y and ends: {xe}x and {ye}y"
            first = False
        else:
            start = end + 1
            xs = start % square_side + 1
            ys = start // square_side + 1

            end = start + smali_dim[name]
            xe = end % square_side + 1
            ye = end // square_side + 1
            image_legend += f"\n{name} starts: {xs}x and {ys}y and ends: {xe}x and {ye}y"

    return image_legend
