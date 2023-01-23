import logging
import os
from ext_tools.DexWave.lib.perturbation_interface import IPerturbation
from ext_tools.DexWave.lib.obfuscation import Obfuscation
from fileinput import FileInput

class Test(IPerturbation):
  def __init__(self):
    super().__init__()
  
  def perturbate(self, obfuscation: Obfuscation):
    self.logger.debug('SMALI recompilation of {} files'.format(len(obfuscation.smali_files)))

    with FileInput(obfuscation.smali_files, inplace=True) as file:
      for line in file:
        print(line, end='')