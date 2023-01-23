#!/usr/bin/env python3
import os
import colorlog
import logging
import numpy as np
from utils.handle_modes import process_path
import tensorflow as tf
from utils.generic_utils import print_log
from ext_tools.DexWave.lib.obfuscation import Obfuscation
from ext_tools.DexWave.lib.mirror import Mirror
from ext_tools.DexWave.lib.perturbations_manager import PerturbationsManager
from ext_tools.DexWave.lib.perturbation_interface import IPerturbation

class DexWave:

  def __init__(self):
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
      "[%(log_color)s%(levelname)s%(reset)s] %(message)s",
      reset=True,
      log_colors={
        'DEBUG':    'yellow',
        'INFO':     'green',
        'WARNING':  'orange',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
      },
      style="%"
    ))

    logging.getLogger().addHandler(handler)
    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.DEBUG)

  def attack_model(
          self,
          input_dex_path,
          output_dex_path,
          model,
          class_info,
          perturbations_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'perturbations'),
          target_class=None
  ):

    mirrorPy = Mirror()

    perturbation_manager = PerturbationsManager(perturbations_path)
    try:

      loaded_perturbations = perturbation_manager.get_all_perturbations()
      self.logger.info(f"perturbations loaded: {','.join(perturbation.name for perturbation in loaded_perturbations)}")
      print_log(f"perturbations loaded: {','.join(perturbation.name for perturbation in loaded_perturbations)}")

      stats = {'succ': [], 'fail': []}

      for outputClasses in os.listdir(input_dex_path):
        for dex_file in os.listdir(f"{input_dex_path}/{outputClasses}"):

          input_dex = f"{input_dex_path}/{outputClasses}/{dex_file}"
          success = False

          obfuscation = Obfuscation(input_dex, output_dex_path)
          obfuscation.inflate()

          # TODO: move to working dir instead
          original_dex_image_path = mirrorPy.elaborate(input_dex, obfuscation.output_dex_dir)
          os.rename(original_dex_image_path, f"{original_dex_image_path[:-4]}ORIGINAL.png")
          original_dex_image_path = f"{original_dex_image_path[:-4]}ORIGINAL.png"
          self.logger.info('original dex image can be found at {}'.format(original_dex_image_path))
          print_log(f'original dex image can be found at {original_dex_image_path}')

          image, _ = process_path(original_dex_image_path)
          image = tf.expand_dims(image, 0)
          prediction = model.predict(image, steps=1)
          original_class = np.argmax(prediction[0])
          original_class_name = f"{class_info['class_names'][int(original_class)]}"

          self.logger.info(f'original dex image has been classified as: {original_class_name}')
          print_log(f'original dex image has been classified as: {original_class_name}')

          for perturbation in loaded_perturbations:
            if not success:
              perturbation.plugin_object.perturbate(obfuscation)

              obfuscation.produce_dex()
              # TODO: move to working dir instead
              obfuscated_dex_image_path = mirrorPy.elaborate(obfuscation.output_dex_path, obfuscation.output_dex_dir)

              image, _ = process_path(obfuscated_dex_image_path)
              image = tf.expand_dims(image, 0)
              new_prediction = model.predict(image, steps=1)
              new_class = np.argmax(new_prediction[0])
              new_class_name = f"{class_info['class_names'][int(new_class)]}"

              if new_class != original_class:

                # Untargeted missclassification, every missclassified class is fine
                if target_class is None:
                  self.logger.debug(
                    f'untargeted misclassification successful via {perturbation.name}: '
                    f'{original_class_name} -> {new_class_name}')
                  print_log(f'untargeted misclassification successful via {perturbation.name}: '
                            f'{original_class_name} -> {new_class_name}')
                  success = True

                # Targeted missclassification successfull, only when missclassifying to a specific target class (i.e. Trusted)
                elif target_class == new_class:
                  self.logger.debug(f'targeted misclassification successful via {perturbation.name}: '
                                    f'{original_class_name} -> {new_class_name}')
                  print_log(f'targeted misclassification successful via {perturbation.name}: '
                            f'{original_class_name} -> {new_class_name}')
                  success = True

                else:
                  self.logger.debug(f'{perturbation.name} perturbation is not strong enough: {new_class_name}')
                  print_log(f'{perturbation.name} perturbation is not strong enough: {new_class_name}')

              else:
                self.logger.debug(f'{perturbation.name} perturbation is not strong enough: {new_class_name}')
                print_log(f'{perturbation.name} perturbation is not strong enough: {new_class_name}')

          if success:
            self.logger.info(f'successful misclassified dex at {obfuscation.output_dex_path} recognized as {new_class}')
            print_log(f'successful misclassified dex at {obfuscation.output_dex_path} recognized as {new_class}')
            stats['succ'].append(obfuscation.output_dex_path)
          else:
            self.logger.error('miscassification failed, you need to write a stronger perturbation')
            print_log('miscassification failed, you need to write a stronger perturbation')
            stats['fail'].append(original_dex_image_path)

          self.logger.info(f"PARTIAL RESULTS: succ {len(stats['succ'])} out of {len(stats['succ']) + len(stats['fail'])}")

      print_log(f"FINAL RESULTS: success on {len(stats['succ'])} samples out of {len(stats['succ']) + len(stats['fail'])}")
      print_log(f"List success : {stats['succ']} ")
      print_log(f"List failed : {stats['fail']} ")

    except Exception as e:
      self.logger.error(e)
      exit(1)

  def classificator_label_parse(label_string: str):
    return label_string.replace(' ', '').split(',')