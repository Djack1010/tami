import os
import cati.utils.tools as tls
from cati.utils.cati_config import DATASETS, DECOMPILED, RESULTS, timeExec
from PIL import Image


def create_dataset(apks, name, side, training_per, validation_per):
    tls.create_folder(f"{DATASETS}/{name}_{timeExec}")
    dataset_path = f"{DATASETS}/{name}_{timeExec}"
    tls.create_folder(f"{dataset_path}/training")
    tls.create_folder(f"{dataset_path}/training/train")
    tls.create_folder(f"{dataset_path}/training/val")
    tls.create_folder(f"{dataset_path}/test")
    for family in apks:
        tls.create_folder(f"{dataset_path}/training/val/{family}")
        tls.create_folder(f"{dataset_path}/training/train/{family}")
        tls.create_folder(f"{dataset_path}/test/{family}")
        training = apks[family] * training_per / 100
        validation = training * validation_per / 100
        training -= validation
        i = 1
        for file in os.listdir(f"{DECOMPILED}/{family}"):
            img = Image.open(f"{RESULTS}/{family}/{file}.png")
            wpercent = (side / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((side, hsize), Image.ANTIALIAS)
            if i < validation:
                img.save(f"{dataset_path}/training/val/{family}/{file}.png")
            elif i < training:
                img.save(f"{dataset_path}/training/train/{family}/{file}.png")
            else:
                img.save(f"{dataset_path}/training/train/{family}/{file}.png")
            i += 1
