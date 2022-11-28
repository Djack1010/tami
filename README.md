# Tami

TAMI (Tool for Analyzing Malware represented as Images) gathers together the code, tools, and approaches presented 
in some publications by Giacomo Iadarola, a PhD student at IIT-CNR and University of Pisa. 

If you are using this repository, please consider [**citing our works**](#publications) (see links at the end of this README file).

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes. We highly suggest running Tami in a [Docker container](#run_docker), especially to run experiments on the GPU.
Otherwise, you can run Tami in a virtualenv (see Section [virtualenv](#virtualenv)).

### Run in a Docker container 
<a name="run_docker"></a>
> SUGGESTED installation, almost mandatory for experimenting on GPU

You can run TAMI in a container built upon the `tensorflow/tensorflow:2.7.0-gpu` image. This is strongly suggested 
for handling dependencies related to GPU drivers because you only need to install 
[Docker](https://docs.docker.com/install/) and the [NVIDIA Docker support](https://github.com/NVIDIA/nvidia-docker) to 
work with the Tensorflow GPU support (see also [Tensorflow Docker Requirements](https://www.tensorflow.org/install/docker) 
for further instructions).

You can either (1. **Suggested**) [download](download) the latest image from our cloud or (2.) [build](#build) Tami image locally.

#### Download latest Tami from the cloud
<a name="download"></a>

In the `docker/` folder of this repository, there is a script `download_and_load_image.sh` which downloads the latest 
Tami image from the cloud and load locally the image in docker. Once loaded, you can run it with `run_container.sh`.

Scripts Usage:
> 
> download_and_load_image.sh
> 
> run_container.sh [--no-gpu] [--quantum]

```
# DEFAULT EXECUTION
docker/download_and_load_image.sh
docker/run_container.sh
```

#### Build Tami locally
<a name="build"></a>

In the `docker/` folder of this repository, there is a Dockerfile that builds the image and installs the requirements 
for TAMI, and two scripts (`build_image.sh` and `run_container.sh`) to handle the docker operations.

Scripts Usage:
> 
> build_image.sh [--quantum]
> 
> run_container.sh [--no-gpu] [--quantum]

```
# DEFAULT EXECUTION
docker/build_image.sh
docker/run_container.sh
```

##### External tools required for vectorization:

[GIST DESCRIPTOR](https://github.com/tuttieee/lear-gist-python)

The script `install.sh` should take care of the gist descriptor tool integration. If something fails, manually install
the repo:
```
git clone https://github.com/tuttieee/lear-gist-python
```

### Run in a Virtualenv
<a name="virtualenv"></a>

##### Tested on Ubuntu 20.04

You can run the script `install.sh` to set up all the necessary dependencies (excluding the GPU ones).
Then, you should install all the necessary libraries with `pip`
```
pip install -r requirements/partial_requirements.txt 
```

---

## Usage

There are 2 scripts that handle Tami executions: `train_test.py` and `post_processing.py`. There are more utilities
scripts in the `scripts` folder, such as backup data and cleaning up old results/logs.

### Train and test models

The script `train_test.py` allows training different DL models over a provided dataset. Also, it allows performing model
assessment (through tuning the hyperparameters), save and load trained models, and output graphs and results on the 
training phase.

See further information on the arguments required with:
```
python train_test.py --help
usage: python train_test.py [-h] -m {DATA,LE_NET,ALEX_NET,STANDARD_CNN,STANDARD_MLP,CUSTOM_CNN,VGG16,VGG19,Inception,ResNet50,MobileNet,DenseNet,EfficientNet,QCNN} -d DATASET
                            [-o OUTPUT_MODEL] [-l LOAD_MODEL] [-t {hyperband,random,bayesian}] [-e EPOCHS] [-b BATCH_SIZE] [-i IMAGE_SIZE] [-w WEIGHTS] [-r LEARNING_RATE]
                            [--mode {train,train-val,train-test,test,gradcam-only}] [-v] [--exclude_top] [--no-caching] [--no-classes]

Tool for Analyzing Malware represented as Images

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -m {DATA,LE_NET,ALEX_NET,STANDARD_CNN,STANDARD_MLP,CUSTOM_CNN,VGG16,VGG19,Inception,ResNet50,MobileNet,DenseNet,EfficientNet,QCNN}, --model {DATA,LE_NET,ALEX_NET,STANDARD_CNN,STANDARD_MLP,CUSTOM_CNN,VGG16,VGG19,Inception,ResNet50,MobileNet,DenseNet,EfficientNet,QCNN}
                        Choose the model to use between the ones implemented
  -d DATASET, --dataset DATASET
                        the dataset path, must have the folder structure: training/train, training/val and test,in each of this folders, one folder per class (see dataset_test)
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
                        FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and channel of the images in input (reshape will be applied)
  -w WEIGHTS, --weights WEIGHTS
                        If you do not want random initialization of the model weights (ex. 'imagenet' or path to weights to be loaded), not available for all models!
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for training models
  --mode {train,train-val,train-test,test,gradcam-only}
                        Choose which mode run between 'train-val' (default), 'train-test', 'train', 'test'.The 'train-val' mode will run a phase of training and validation on the
                        training and validation set, the 'train-test' mode will run a phase of training on the training+validation sets and then test on the test set, the 'train'
                        mode will run only a phase of training on the training+validation sets, the 'test' mode will run only a phase of test on the test set. The 'gradcam' has been
                        moved to 'post_processing.py'
  -v, --version         show program's version number and exit
  --exclude_top         Exclude the fully-connected layer at the top of the network (default INCLUDE)
  --no-caching          Caching dataset on file and loading per batches (IF db too big for memory)
  --no-classes          In case of mode including test, skip results for each class (only cumulative results)
```

Logs, figure and performance results are stored in the `results` and `tuning` folders.
Tensorboard can be used to print graph of training and validation trend.
```
tensorboard --logdir results/tensorboard/fit/
```

### Analyze the results

The script `post_processing.py` performs operations and analysis over result sets of a previous training phase.
In detail, it applies the Grad-cam on a loaded (and trained) model, and then it can also run IF/IM-SSIM analysis on the
generated heatmaps.

See further information on the arguments required with:
```
python post_processing.py --help
usage: python post_processing.py [-h] [-l LOAD_MODEL] [-d DATASET] [-gl SAMPLE_GRADCAM] [-gs SHAPE_GRADCAM] [-sf [SSIM_FOLDERS [SSIM_FOLDERS ...]]]
                                 [--mode {IFIM-SSIM,gradcam-only,gradcam-cati}] [-v] [--include_all]

Tool for Analyzing Malware represented as Images

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -l LOAD_MODEL, --load_model LOAD_MODEL
                        Name of model to load
  -d DATASET, --dataset DATASET
                        the dataset path, must have the folder structure: training/train, training/val and test,in each of this folders, one folder per class (see dataset_test)
  -gl SAMPLE_GRADCAM, --sample_gradcam SAMPLE_GRADCAM
                        Limit gradcam to X samples randomly extracted from the test set
  -gs SHAPE_GRADCAM, --shape_gradcam SHAPE_GRADCAM
                        Select gradcam target layer with at least shapeXshape (for comparing different models)
  -sf [SSIM_FOLDERS [SSIM_FOLDERS ...]], --ssim_folders [SSIM_FOLDERS [SSIM_FOLDERS ...]]
                        List of gradcam results folder to compare with IF-SSIM and IM-SSIM
  --mode {IFIM-SSIM,gradcam-only,gradcam-cati}
                        Choose which mode run between 'gradcam-only' (default), 'gradcam-cati', 'IFIM-SSIM'The 'gradcam-[cati|only]' will run the gradcam analysis on the model
                        provided. 'gradcam-only' will generate the heatmaps only, while 'gradcam-cati will also run the cati tool to reverse process and select the code from the
                        heatmap to the decompiled smali (if provided, see cati README)
  -v, --version         show program's version number and exit
  --include_all         Include all possible heatmaps in the IFIM-SSIM analysis (default, choose a random subset)
```

## Authors & References

* **Giacomo Iadarola** - *main contributor* - [Djack1010](https://github.com/Djack1010) giacomo.iadarola(at)iit.cnr.it
* **Christian Peluso** - *cati tool* - [1Stohk1](https://github.com/1Stohk1)
* **Francesco Mercaldo** - *contributor* - [FrancescoMercaldo](https://github.com/FrancescoMercaldo)
* **Fabrizio Ravelli** - *contributor* - [reFraw](https://github.com/reFraw)

<a name="publications"></a>
If you are using this repository, please cite our work by referring to our publications (BibTex format):
```
@inproceedings{iadarola2021semi,
  title={A Semi-Automated Explainability-Driven Approach for Malware Analysis through Deep Learning},
  author={Iadarola, Giacomo and Casolare, Rosangela and Martinelli, Fabio and Mercaldo, Francesco and Peluso, Christian and Santone, Antonella},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}

@inproceedings{gerardi2021perturbation,
  title={Perturbation of Image-based Malware Detection with Smali level morphing techniques},
  author={Gerardi, Federico and Iadarola, Giacomo and Martinelli, Fabio and Santone, Antonella and Mercaldo, Francesco},
  booktitle={2021 IEEE Intl Conf on Parallel \& Distributed Processing with Applications, Big Data \& Cloud Computing, Sustainable Computing \& Communications, Social Computing \& Networking (ISPA/BDCloud/SocialCom/SustainCom)},
  pages={1651--1656},
  year={2021},
  organization={IEEE}
}

@article{iadarola2021towards,
  title={Towards an Interpretable Deep Learning Model for Mobile Malware Detection and Family Identification},
  author={Iadarola, Giacomo and Martinelli, Fabio and Mercaldo, Francesco and Santone, Antonella},
  journal={Computers \& Security},
  pages={102198},
  year={2021},
  publisher={Elsevier}
}
```

### Sub-repositories

List of other repositories related to this one, specifically created for a project/work/paper and containing only a subset of files, the necessary ones.

* [Towards Interpretable DL Models](https://github.com/Djack1010/towards_interpretable_DL_models)
* [Semi-Automated Explainability-Driven Approach for Malware Analysis](https://github.com/Djack1010/malware_img2smali)
* [Perturbation of Image-based Malware Detectionwith Smali level morphing techniques](https://github.com/AzraelSec/nedo)

#### Acknowledgements

The authors would like to thank the 'Trust, Security and Privacy' research group within the 
[Institute of Informatics and Telematics](https://www.iit.cnr.it/) (CNR - Pisa, Italy), which support their research.

The Grad-CAM code is based on the work of Adrian Rosebrock, available 
[here](https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/).
