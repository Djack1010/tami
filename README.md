# Tami

TAMI (Tool for Analyzing Malware represented as Images) implements the approach presented in some publication by 
Giacomo Iadarola, a PhD student at IIT-CNR and University of Pisa. If you are using this repository, please **cite our 
works** (see links at the end of this README file).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Dependencies

##### Ubuntu 18.04

The project needs Python3 to be run, and it has been tested in Linux Environment (Ubuntu 18.04).
It also needs Tensorflow 2.1, the dependencies for training on the GPU and installing all the requirements in 
`requirements/requirements_ubuntu18.txt`. You also need to set the variable `main_path` in utils/config.py to the full 
path to the repository folder on your local machine.

##### Ubuntu 20.04

It was tested also on Ubuntu 20.04 with Tensorflow 2.3. The new requirements are stored in 
`requirements/requirements_ubuntu20.txt`. You can also run the script `install.sh` to set up all the necessary 
dependencies (excluding the GPU ones).

#### External tools required for vectorization:
GIST DESCRIPTOR
```
git clone https://github.com/tuttieee/lear-gist-python
```

#### Usage

The DL models can be run with two different scripts:
* `main_base.py` which uses the models in `models_base` package. They are the basic ones and exploit the Keras function
for training, validation and test.
* `main_impl.py` which uses the models in `models_impl` package, where the model structure and also the training,
validation and test functions are customized, thus can be changed and tested for educational and experimenting purposes.

See further information on the arguments required with:
```
python main_[base|impl].py --help
```

## Authors & References

* **Giacomo Iadarola** - *main contributor* - [Djack1010](https://github.com/Djack1010) giacomo.iadarola(at)iit.cnr.it

If you are using this repository, please cite our work by referring to our publications (BibTex format):
```
WORK IN PROGRESS...
```
