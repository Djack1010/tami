import os
import cati.utils.tools as tls
from tqdm import tqdm
from cati.utils.cati_config import DATASETS, DECOMPILED, RESULTS, timeExec
from PIL import Image


def create_dataset(apks, name, side, training_per, validation_per):
    tls.create_folder(f"{DATASETS}/{name}_{timeExec}")
    dataset_path = f"{DATASETS}/{name}_{timeExec}"
    tls.create_folder(f"{dataset_path}/test")
    tls.create_folder(f"{dataset_path}/training")
    tls.create_folder(f"{dataset_path}/training/train")
    tls.create_folder(f"{dataset_path}/training/val")
    for family in apks:
        tls.create_folder(f"{dataset_path}/training/val/{family}")
        tls.create_folder(f"{dataset_path}/training/train/{family}")
        tls.create_folder(f"{dataset_path}/test/{family}")
        training = apks[family]['num'] * training_per / 100
        validation = training * validation_per / 100
        # training -= validation
        i = 1
        image_progressing = tqdm(os.listdir(f"{DECOMPILED}/{family}"),
                                 position=0, unit=' image', bar_format='{desc}|{bar:20}{r_bar}', leave=False)
        for file in image_progressing:
            image_progressing.set_description(f'Moving the file ({file[0:10]}...) of the folder ({family})')
            img = Image.open(f"{RESULTS}/{name}/{family}/{file}.png")
            wpercent = (side / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((side, hsize), Image.ANTIALIAS)
            if i < validation:
                img.save(f"{dataset_path}/training/val/{family}/{file}.png")
            elif i < training:
                img.save(f"{dataset_path}/training/train/{family}/{file}.png")
            else:
                img.save(f"{dataset_path}/test/{family}/{file}.png")
            i += 1
