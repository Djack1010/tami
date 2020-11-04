# Tami

TAMI (Tool for Analyzing Malware represented as Images) implements the approach presented in some publication by Giacomo Iadarola, Ph.D. student at IIT-CNR and University of Pisa.
If you are using this repository, please cite our works (see links at the end of this README file).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Dependencies

##### Ubuntu 18.04

The project needs Python3 to be run, and it has been tested in Linux Environment (Ubuntu 18.04).
It also needs Tensorflow 2.1, the dependencies for training on the GPU and installing all the requirmentes in `requirements.txt`.
You also need to set the variable 'main_path' in utils/config.py to the full path to the repository folder on your local machine.

##### Ubuntu 20.04

It was tested also on Ubuntu 20.04 with Tensorflow 2.3. The new requirements are stored in `requirements_ubuntu20.txt`.
You can also run the script `install.sh` to setup all the necessary dependencies (excludin the GPU ones).

#### External tools required for vectorization:
GIST DESCRIPTOR
```
git clone https://github.com/tuttieee/lear-gist-python
```

## Authors & References

* **Giacomo Iadarola** - *main contributor* - [Djack1010](https://github.com/Djack1010) giacomo.iadarola(at)iit.cnr.it

If you are using this repository, please cite our work by refering to our publications (bibtex format):
```
WORK IN PROGRESS...
```
