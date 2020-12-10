# Cati

CATI (Converter Apk To Image) is a project implemented by Christian Peluso, student of Information Technology in Pesche 
(IS).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes.

1. Use the command git clone in your system 

2. Create a virtual environment with Python, then activate the environment and install the dependencies reported in the 
requirements file.

3. Install [apktool](https://ibotpeaches.github.io/Apktool/install) and follow the install instructions to set up 
appropriately.

4. Use the script `install.sh` to set up the current path to the project itself.

5. Move your apk files in the folder "sample", be sure to put the apks in folders named after their macro type, like this:

```
tree /cati/sample/
|-- type1
|   |-- application.apk
|   |-- application2.apk
|   `-- application3.apk
|-- type2
|   |-- app.apk
|   |-- app2.apk
|   `-- app3.apk
`-- type3
    |-- soft.apk
    |-- soft2.apk
    `-- soft3.apk
```

6. Once there you can run `python apk_decompiler.py` to extract the smali files.

7. Finally, you can run `python main_cati.py` that will convert the smali files in OPCode and then convert them in 
PNGs.

### Dependencies

The project needs Python3 to be run, it has been tested in Windows 10, Ubuntu and Manjaro KDE.

#### External tools required for vectorization:

- [GIST DESCRIPTOR](https://github.com/tuttieee/lear-gist-python)
- [APKTOOL](https://ibotpeaches.github.io/Apktool)

#### Usage

The datasets can be created with the main_cati.py script:

See further information on the arguments required with:

```

python main_cati.py --help
usage: main_cati.py [-h] [-o DATASET_NAME] [-t TRAINING] [-v VALIDATION] [-i IMAGE_SIZE] [-b BATCH_SIZE] 
                    [-i IMAGE_SIZE][--no_storage] [--no_results]
               
  -t TRAINING, --training TRAINING
                        Percentage of data to be saved in training, insert a number between 10 to 90
  -v VALIDATION, --validation VALIDATION
                        Percentage of data to be saved in validation, insert a number between 10 to 90
  -i IMAGE_SIZE, --image_size IMAGE_SIZE
                        FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and channel of the images in input default is 
                        [250x1] (reshape will be applied)
  -o OUTPUT_NAME, --output_name OUTPUT_NAME
                        Enter the name by which you want to call the output
  --no_storage          Do not create a dataset in tami/DATASETS folder
  --no_results          Do nothing in results folder, so no creation of legends or images of the smali files in
                        cati/results folder


```
