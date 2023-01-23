#!/usr/bin/env python3
from ext_tools.DexWave.lib.dexwave import DexWave
from argparse import ArgumentParser
from dotenv import load_dotenv
import os

if __name__ == '__main__':
  dexWave = DexWave()
  dexWave.logging_setup()

  parser = ArgumentParser()
  parser.add_argument('input_dex_path', type=str, help='path of the input dex file')
  parser.add_argument('output_dex_path', type=str, help='path of the output obfuscated dex file')
  parser.add_argument('--weights_path', type=str, help='path of the tensorflow model weights', default=None)
  parser.add_argument('--classificator_labels', type=str, help='classificator labels string divided by commas', default=None)
  args = parser.parse_args()

  load_dotenv(os.path.join(os.path.realpath(os.path.dirname(__file__)), 'resources', '.wave'), verbose=True)

  weights_path = os.getenv('CLASSIFICATOR_WEIGHTS_PATH') or args.weights_path
  classificator_labels = os.getenv('CLASSIFICATOR_LABELS') or args.classificator_labels

  dexWave.attack(args.input_dex_path, args.output_dex_path, weights_path, classificator_labels)