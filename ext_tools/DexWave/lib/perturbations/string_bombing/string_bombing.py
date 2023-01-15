#!/usr/bin/python3
from perturbation_interface import IPerturbation
from obfuscation import Obfuscation
from fileinput import FileInput
import secrets
import re
import os

class StringBombing(IPerturbation):
    def __init__(self):
      super().__init__()
    
    def perturbate(self, obfuscation: Obfuscation):
      pattern = re.compile(r"\s+const-string v(?P<register_val>\d)\S+")
      register_patched = 0
      with FileInput(obfuscation.smali_files, inplace=True) as file:
        for line in file:
          match = pattern.match(line)
          if match:
            register = match.group('register_val').strip()
            register_patched = register_patched + 1
            print("\tconst-string v{}, \"{}\"".format(register, self.get_random_token()))
          print(line, end='')
      self.logger.debug('{} registers overloaded by StringBombing'.format(register_patched))
    
    def get_random_token(self):
      return secrets.token_hex(10)

          
