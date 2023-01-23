#!/usr/bin/env python3
import logging
import os
import secrets
from .smali_api import SmaliAPI

class Obfuscation:
  # input_dex_path: directory path
  # output_dex_name: obfuscated dex name
  def __init__(self, input_dex_path, output_dex_path):
    self.input_dex_path = input_dex_path
    
    self.output_dex_dir = output_dex_path
    tmp_dex_name = (os.path.basename(input_dex_path) or 'DexWaved.dex')
    self.output_dex_name = tmp_dex_name if tmp_dex_name.endswith('.dex') else '{}.dex'.format(tmp_dex_name)
    self.output_dex_path = os.path.join(self.output_dex_dir, self.output_dex_name)
    
    self.working_dir = secrets.token_hex(10)
    self.working_dir_path = os.path.join('/tmp', self.working_dir)
    
    self.smali_api = SmaliAPI()
    self.smali_files = []

    self.inflated = False

    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.DEBUG)

  def inflate(self):
    if not self.inflated:
      if os.path.isfile(self.input_dex_path):
        if os.path.isdir(self.output_dex_dir):
          self.logger.info('file {} is going to be obfuscated into {}'.format(self.input_dex_path, self.output_dex_path))
          
          self.smali_api.baksmali(self.input_dex_path, '/tmp', self.working_dir)

          collected_files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(self.working_dir_path)
            for file in files
            if str(file).endswith('.smali')]
          
          filtered_files = [
            file for file in collected_files
            if not any(
              map(
                lambda x: str(os.path.relpath(file, self.working_dir_path)).startswith(x), self.get_to_ignore_libs()
              )
            )
          ]

          self.smali_files = filtered_files
          self.logger.debug('{} collected files'.format(len(filtered_files)))
        else:
          raise OSError('directory {} is unavailable'.format(self.output_dex_dir))
      else:
        raise FileNotFoundError('file {} does not exists or is unaccessible'.format(self.input_dex_path))

  def produce_dex(self):
    if self.inflate:
      if os.path.isdir(self.working_dir_path):
        self.smali_api.smali(self.working_dir_path, self.output_dex_path)
        self.logger.info('obfuscated file created at {}'.format(self.output_dex_path))
      else:
        raise OSError('smali directory {} unavailable'.format(self.working_dir_path))
    else:
      raise Exception('obfuscation not inflated yet')

  def get_to_ignore_libs(self):
    try:
      with open(os.path.join(os.path.realpath(os.path.dirname(__file__)), 'resources', 'libs_to_ignore.txt')) as lines:
        return [str(line).strip() for line in lines if str(line).strip() != '']
    except Exception as e:
      raise Exception('unable to read the libs_to_ignore.txt file at {}: {}'.format(os.path.join(os.path.realpath(os.path.dirname(__file__)), 'resources', 'libs_to_ignore.txt'), e))