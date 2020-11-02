import math
import random

from PIL import Image


def rgb_image_generator(num):
    dim = int(math.sqrt(num)) + 1
    img = Image.new('RGB', (dim, dim), 'white')
    pix_map = img.load()
    return img, pix_map, dim


def pixel_generator(smali_dim, pixMap, dim):
    x, y = 0, 0  # coordinate pixelmap
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
                print("errore dimensione immagine")
                break
            i += 1

    return pixMap


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
