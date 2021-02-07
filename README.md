# Tami

TAMI (Tool for Analyzing Malware represented as Images) gathers toghether the code, tools and approaches presented in some publication by 
Giacomo Iadarola, a PhD student at IIT-CNR and University of Pisa. 

If you are using this repository, please consider [**citing our works**](#publications) (see links at the end of this README file).

### Sub-repositories

List of other repositories related to this one, specifically created for a project/work/paper and containing only a subset of files, the necessary ones.

* [Towards Interpretable DL Models](https://github.com/Djack1010/towards_interpretable_DL_models)
* [Semi-Automated Explainability-Driven Approach for Malware Analysis](https://github.com/Djack1010/malware_img2smali)

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
usage: main.py [-h] -m {DATA,BASIC_CNN,BASIC_LSTM,BASIC_MLP,NEDO,VINC,VGG16} -d DATASET [-o OUTPUT_MODEL] 
               [-l LOAD_MODEL] [-t {hyperband,random,bayesian}] [-e EPOCHS] [-b BATCH_SIZE] [-i IMAGE_SIZE] 
               [-w WEIGHTS] [--mode {train-val,train-test,test,gradcam-cati,gradcam-only}] [--exclude_top] 
               [--caching]

Deep Learning Image-based Malware Classification

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -m {DATA,BASIC_CNN,BASIC_LSTM,BASIC_MLP,NEDO,VINC,VGG16}, --model {DATA,BASIC_CNN,BASIC_LSTM,BASIC_MLP,NEDO,VINC,VGG16}
                        Choose the model to use between the ones implemented
  -d DATASET, --dataset DATASET
                        the dataset path, must have the folder structure: training/train, training/val and test,
                        in each of this folders, one folder per class (see dataset_test)
  -o OUTPUT_MODEL, --output_model OUTPUT_MODEL
                        Name of model to store
  -l LOAD_MODEL, --load_model LOAD_MODEL
                        Name of model to load
  -t {hyperband,random,bayesian}, --tuning {hyperband,random,bayesian}
                        Run Keras Tuner for tuning hyperparameters, options: [hyperband, random, bayesian]
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -i IMAGE_SIZE, --image_size IMAGE_SIZE
                        FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and channel of the images in input 
                        (reshape will be applied)
  -w WEIGHTS, --weights WEIGHTS
                        If you do not want random initialization of the model weights (ex. 'imagenet' or path 
                        to weights to be loaded), not available for all models!
  --mode {train-val,train-test,test,gradcam-cati,gradcam-only}
                        Choose which mode run between 'train-val' (default), 'train-test', 'test' or 'gradcam'. 
                        The 'train-val' mode will run a phase of training and validation on the training and 
                        validation set, the 'train-test' mode will run a phase of training on the training+validation 
                        sets and then test on the test set, the 'test' mode will run only a phase of test on 
                        the test set. The 'gradcam-[cati|only]' will run the gradcam analysis on the model provided. 
                        'gradcam-only' will generate the heatmaps only, while 'gradcam-cati will also run 
                        the cati tool to reverse process and select the code from the heatmap to the decompiled 
                        smali (if provided, see cati README)
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
* **Christian Peluso** - *cati tool* - [1Stohk1](https://github.com/1Stohk1)

<a name="publications"></a>
If you are using this repository, please cite our work by referring to our publications (BibTex format):
```
@article{iadarola2021towards,
  title={Towards an Interpretable Deep Learning Model for Mobile Malware Detection and Family Identification},
  author={Iadarola, Giacomo and Martinelli, Fabio and Mercaldo, Francesco and Santone, Antonella},
  journal={Computers \& Security},
  pages={102198},
  year={2021},
  publisher={Elsevier}
}

@inproceedings{iadarola2020evaluating,
  title={Evaluating Deep Learning Classification Reliability in Android Malware Family Detection},
  author={Iadarola, Giacomo and Martinelli, Fabio and Mercaldo, Francesco and Santone, Antonella},
  booktitle={2020 IEEE International Symposium on Software Reliability Engineering Workshops (ISSREW)},
  pages={255--260},
  year={2020},
  organization={IEEE}
}
```

#### Acknowledgements

The authors would like to thank the 'Trust, Security and Privacy' research group within the [Institute of Informatics and Telematics](https://www.iit.cnr.it/) (CNR - Pisa, Italy), that support their researches.

The Grad-CAM code is based on the work of Adrian Rosebrock available [here](https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/).
