import math
import os
from queue import Queue
from threading import Thread
from utils.generic_utils import print_log

from PIL import Image


def getBinaryData(filename):
    """
    Extract byte values from binary executable file and store them into list
    :param filename: executable file name
    :return: byte value list
    """

    binary_values = []

    with open(filename, 'rb') as fileobject:
        # read file byte by byte
        data = fileobject.read(1)

        while data != b'':
            binary_values.append(ord(data))
            data = fileobject.read(1)

    return binary_values


def createGreyScaleImage(filename, width=None):
    """
    Create greyscale image from binary data. Use given with if defined or create square size image from binary data.
    :param filename: image filename
    """
    greyscale_data = getBinaryData(filename)
    size = get_size(len(greyscale_data), width)
    save_file(filename, greyscale_data, size, 'L')


def createRGBImage(filename, width=None):
    """
    Create RGB image from 24 bit binary data 8bit Red, 8 bit Green, 8bit Blue
    :param filename: image filename
    """
    index = 0
    rgb_data = []

    # Read binary file
    binary_data = getBinaryData(filename)

    # Create R,G,B pixels
    while (index + 3) < len(binary_data):
        R = binary_data[index]
        G = binary_data[index + 1]
        B = binary_data[index + 2]
        index += 3
        rgb_data.append((R, G, B))

    size = get_size(len(rgb_data), width)
    save_file(filename, rgb_data, size, 'RGB')


def save_file(filename, data, size, image_type):
    try:
        image = Image.new(image_type, size)
        image.putdata(data)

        # setup output filename
        dataset_name = None  # Extract name of original dataset
        if '/training/' in filename:
            dataset_name = filename.split('/training/')[0].split('/')[-1]
        elif '/test/' in filename:
            dataset_name = filename.split('/test/')[0].split('/')[-1]
        else:
            print_log(f"ERROR! /training/ or /test/ folders not found in '{filename}', cannot convert file",
                      print_on_screen=True)
            return
        # Replace name of the original dataset with 'originalname_imagetype'
        dirname = os.path.dirname(filename).replace(dataset_name,
                                                    f"{dataset_name}_{image_type if image_type == 'RGB' else 'GRAY'}")
        name, _ = os.path.splitext(filename)
        name = os.path.basename(name)
        imagename = dirname + os.sep + name + '.png'
        os.makedirs(os.path.dirname(imagename), exist_ok=True)

        image.save(imagename)
        print_log(f"File converted: {imagename}", print_on_screen=True)
    except Exception as err:
        print_log(f"ERROR! {str(err)}", print_on_screen=True)


def get_size(data_length, width=None):
    # source Malware images: visualization and automatic classification by L. Nataraj
    # url : http://dl.acm.org/citation.cfm?id=2016908

    if width is None:  # with don't specified any with value

        size = data_length

        if size < 10240:
            width = 32
        elif 10240 <= size <= 10240 * 3:
            width = 64
        elif 10240 * 3 <= size <= 10240 * 6:
            width = 128
        elif 10240 * 6 <= size <= 10240 * 10:
            width = 256
        elif 10240 * 10 <= size <= 10240 * 20:
            width = 384
        elif 10240 * 20 <= size <= 10240 * 50:
            width = 512
        elif 10240 * 50 <= size <= 10240 * 100:
            width = 768
        else:
            width = 1024

        height = int(size / width) + 1

    else:
        width = int(math.sqrt(data_length)) + 1
        height = width

    return width, height


def run(file_queue, width, mode):
    while not file_queue.empty():
        filename = file_queue.get()
        if 'gray' in mode:
            createGreyScaleImage(filename, width)
        if 'rgb' in mode:
            createRGBImage(filename, width)
        file_queue.task_done()


def binary2image(input_dir, width=None, thread_number=5, mode='rgb-gray'):
    # Get all executable files in input directory and add them into queue
    file_queue = Queue()
    for root, directories, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            # ADD to the queue only files in test and training set (avoid other files in the dataset tree folder)
            if f'/test/' in file_path or '/training/train/' in file_path or '/training/val/' in file_path:
                file_queue.put(file_path)

    if file_queue.qsize() == 0:
        print_log(f"ERROR! No suitable files found in {input_dir}/[training/val|training/train|test] folders. Exiting.",
                  print_on_screen=True)
        exit()

    # Start thread
    for index in range(thread_number):
        thread = Thread(target=run, args=[file_queue, width, mode])
        thread.daemon = True
        thread.start()
    file_queue.join()

