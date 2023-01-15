#!/usr/bin/env python3
import logging
import subprocess
import os

class SmaliAPI:
  def __init__(self):
    self.smali_jar_location = os.getenv('SMALI_PATH') or os.path.join(os.path.realpath(os.path.dirname(__file__)), 'resources', 'smali', 'smali.jar')
    self.baksmali_jar_location = os.getenv('BAKSMALI_PATH') or os.path.join(os.path.realpath(os.path.dirname(__file__)), 'resources', 'smali', 'baksmali.jar')

    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.DEBUG)

    if not os.path.isfile(self.smali_jar_location) or not os.path.isfile(self.baksmali_jar_location):
      self.logger.debug('SMALI_PATH: {}'.format(self.smali_jar_location))
      self.logger.debug('BAKSMALI_PATH: {}'.format(self.baksmali_jar_location))
      raise RuntimeError('smali or baksmali not located')
    else:
      self.logger.debug('SMALI_PATH: {}'.format(self.smali_jar_location))
      self.logger.debug('BAKSMALI_PATH: {}'.format(self.baksmali_jar_location))

  def baksmali(self, intput_dex_path, output_parent_path, output_dir_name):
    if os.path.isfile(intput_dex_path):
      if os.path.isdir(output_parent_path):
        execution_args = [
          'java',
          '-jar',
          self.baksmali_jar_location,
          'd',
          '-o',
          os.path.join(os.path.realpath(output_parent_path), output_dir_name),
          intput_dex_path
        ]
        try:
          subprocess.run(execution_args)
          self.logger.debug('{} dex has been disassembled into {}'.format(intput_dex_path, os.path.join(os.path.realpath(output_parent_path), output_dir_name)))
        except subprocess.CalledProcessError as e:
          raise Exception('unable to execute baksmali: {}'.format(e))
      else:
        raise OSError('path {} does not exist'.format(output_parent_path))
    else:
      raise FileNotFoundError('file {} does not exist'.format(intput_dex_path))

  def smali(self, input_smali_path, output_dex_path):
    if os.path.isdir(input_smali_path):
      execution_args = [
        'java',
        '-jar',
        self.smali_jar_location,
        'a',
        '-o',
        output_dex_path,
        input_smali_path
      ]
      try:
        subprocess.run(execution_args)
        self.logger.debug('{} smali has been assembled into {}'.format(input_smali_path, output_dex_path))
      except subprocess.CalledProcessError as e:
        raise Exception('unable to execute smali: {}'.format(e))
    else:
      raise OSError('path {} does not exist'.format(input_smali_path))