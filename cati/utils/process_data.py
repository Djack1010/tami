from cati.utils.tools import *
from cati.utils.config import *
from PIL import Image


def create_dataset(apks, dims, percentual):
    create_folder(f"{DATASETS}/dataset_{timeExec}")
    dataset_path = f"{DATASETS}/dataset_{timeExec}"
    create_folder(f"{dataset_path}/training")
    create_folder(f"{dataset_path}/training/train")
    create_folder(f"{dataset_path}/training/val")
    create_folder(f"{dataset_path}/test")
    for family in apks:
        create_folder(f"{dataset_path}/training/val/{family}")
        create_folder(f"{dataset_path}/training/train/{family}")
        create_folder(f"{dataset_path}/test/{family}")
        training = apks[family] * percentual / 100
        validation = training * 0.20
        training -= validation
        side = dims
        i = 0
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
