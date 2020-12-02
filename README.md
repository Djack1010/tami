# Tami

TAMI (Tool for Analyzing Malware represented as Images) gathers toghether the code, tools and approaches presented in some publication by 
Giacomo Iadarola, a PhD student at IIT-CNR and University of Pisa. 

If you are using this repository, please consider **citing our 
works** (see links at the end of this README file).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes.

### Dependencies

##### Ubuntu 20.04

You can run the script `install.sh` to set up all the necessary dependencies (excluding the GPU ones).
Then, you should install all the necessary libraries with `pip`
```
pip install -r requirements.txt 
```

#### External tools required for vectorization:

[GIST DESCRIPTOR](https://github.com/tuttieee/lear-gist-python)

The script `install.sh` should take care of the gist descriptor tool integration. If something fails, manually install
the repo:
```
git clone https://github.com/tuttieee/lear-gist-python
```

#### Usage

The DL models can be run with the `main.py` scripts:

See further information on the arguments required with:
```
python main.py --help
usage: main.py [-h] -m MODEL -d DATASET [-o OUTPUT_MODEL] [-l LOAD_MODEL] [-t TUNING] [-e EPOCHS] [-b BATCH_SIZE] [-i IMAGE_SIZE] [-w WEIGHTS] [--mode MODE] [--exclude_top]
               [--caching]

Deep Learning Image-based Malware Classification

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -m MODEL, --model MODEL
                        DATA, BASIC, NEDO or VGG16
  -d DATASET, --dataset DATASET
                        the dataset path, must have the folder structure: training/train, training/val and test,in each of this folders, one folder per class (see
                        dataset_test)
  -o OUTPUT_MODEL, --output_model OUTPUT_MODEL
                        Name of model to store
  -l LOAD_MODEL, --load_model LOAD_MODEL
                        Name of model to load
  -t TUNING, --tuning TUNING
                        Run Keras Tuner for tuning hyperparameters, chose: [hyperband, random, bayesian]
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -i IMAGE_SIZE, --image_size IMAGE_SIZE
                        FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and channel of the images in input (reshape will be applied)
  -w WEIGHTS, --weights WEIGHTS
                        If you do not want random initialization of the model weights (ex. 'imagenet' or path to weights to be loaded), not available for all models!
  --mode MODE           Choose which mode run between 'train-val' (default), 'train-test', 'test' or 'gradcam'. The 'train-val' mode will run a phase of training and
                        validation on the training and validation set, the 'train-test' mode will run a phase of training on the training+validation sets and then test on
                        the test set, the 'test' mode will run only a phase of test on the test set. The 'gradcam' will run the gradcam analysis on the modelprovided.
  --exclude_top         Exclude the fully-connected layer at the top of the network (default INCLUDE)
  --caching             Caching dataset on file and loading per batches (IF db too big for memory)
```

Logs, figure and performance results are stored in `results` and `tuning` folders.
Tensorboard can be used to print graph of training and validation trend.
```
tensorboard --logdir results/tensorboard/fit/
```

### Deprecated 

##### Compatibility with Ubuntu 18.04

_**Update**: All the recent experiments were conducted on Ubuntu 20.04. We do not know it these old settings are still 
working._

The project needs Python3 to be run, and it has been tested in Linux Environment (Ubuntu 18.04).
It also needs Tensorflow 2.1, the dependencies for training on the GPU and installing all the requirements in 
`old_tool/requirements_ubuntu18.txt`. You also need to set the variable `main_path` in `old_tool/utils_backup/config.py`
to the full path to the repository folder on your local machine.

The `old_tool` package contains code and implementation of models from scratch. Right now, it is useful only for 
educational purposes, in order to get into the topic without making use of the Keras API to build the models.

## Authors & References

* **Giacomo Iadarola** - *main contributor* - [Djack1010](https://github.com/Djack1010) giacomo.iadarola(at)iit.cnr.it

If you are using this repository, please cite our work by referring to our publications (BibTex format):
```
WORK IN PROGRESS...
```
