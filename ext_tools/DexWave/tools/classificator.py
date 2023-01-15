#!/usr/bin/env python3
from sys import argv
import logging
from lib.classificator.Model import MCModel
from lib.classificator.MalwareClassificator import MalwareClassificator

def help(name):
    logging.error("usage: {} <image_path> [<model_weights>]".format(name))
    exit()

family_labels = [
    'Airpush_variety1',
    'Dowgin_variety1',
    'FakeInst_variety3',
    'Fusob_variety2',  
    'Mecor_variety1',
    'trusted'
]

if __name__ == '__main__':
    help(argv[0]) if len(argv) < 2 else None

    weigths_path = argv[2] if argv[2] else './weights_result'

    classificator = MalwareClassificator(6, 200, family_labels)

    try:
        if len(argv) >= 3:
            classificator.load_weights(argv[2])
        else:
            classificator.load_weights()
    except IOError as e:
        logging.error(str(e))
        exit()
    
    if not '-q' in argv:
            logging.debug('model correctly loaded')
            logging.debug('loading image {}'.format(argv[1]))
    
    try:
        prediction = classificator.predict(argv[1])
    except Exception as e:
        logging.error(str(e))
        exit()

    logging.info('prediction for the image: {}'.format(classificator.get_label_from_prediction(prediction)))