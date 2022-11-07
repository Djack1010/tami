#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/.."
cd ${SCRIPTPATH}

# TEMPLATE for running/scheduling experiments within container or virtualenv

# MODEL ASSESSMENT, trying different hyper-parameters such as batch_size, epochs, learning rate, input size
# NOTES: --mode [train-val|train-test]
python main.py -d DATASETS/dataset_test_malware/ -m CUSTOM_CNN -i 150x1 -b 32 -e 5 -r 0.1 --mode train-val
python main.py -d DATASETS/dataset_test_malware/ -m CUSTOM_CNN -i 150x1 -b 16 -e 5 -r 0.01 --mode train-val
# You can also use Keras Tuner to schedule experiments on hyper-parameters (see build_tuning of the CUSTOM_CNN model)
python main.py -d DATASETS/dataset_test_malware/ -m CUSTOM_CNN -i 150x1 -t hyperband

# MODEL TRAINING, once we have defined the best hyper-parameters, train and save the model with -o <NAME>
# NOTES: --mode [train|train-test]
python main.py -d DATASETS/dataset_test_malware/ -m CUSTOM_CNN -i 150x1 -b 32 -e 10 -r 0.01 -o test1 --mode train-test

# MODEL TEST, we can upload a trained model with -l <NAME> to perform test or run gradcam on it
# NOTES: --mode [test|gradcam-[only|cati]]
python main.py -d DATASETS/dataset_test_malware/ -m CUSTOM_CNN -i 150x1 -l test1 --mode test
python main.py -d DATASETS/dataset_test_malware/ -m CUSTOM_CNN -i 150x1 -l test1 --mode gradcam-only
