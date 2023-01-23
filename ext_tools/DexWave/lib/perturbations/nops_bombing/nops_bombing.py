#!/usr/bin/env python3
from ext_tools.DexWave.lib.perturbation_interface import IPerturbation
from ext_tools.DexWave.lib.obfuscation import Obfuscation
from fileinput import FileInput
from secrets import choice
import re
import os

class NopsBombing(IPerturbation):
  def __init__(self):
    super().__init__()
  
  def perturbate(self, obfuscation: Obfuscation):
    pattern = re.compile(r"\s+(?P<op_code>\S+)")
    valid_opcodes = self.get_valid_nops()
    patched_opcodes = 0

    with FileInput(obfuscation.smali_files, inplace=True) as file:
      for line in file:
        print(line, end='')
        match = pattern.match(line)
        if match:
          op_code = match.group('op_code').strip()
          if op_code in valid_opcodes:
            patched_opcodes = patched_opcodes + 1
            for i in range(1, choice(range(1, 10))):
              print("\tnop")
    
    self.logger.debug('{} op_codes patched by NopsBombing'.format(patched_opcodes))

  def get_valid_nops(self):
    with open(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'nop_valid_op_codes.txt'))) as nop_file:
      return [s.strip() for s in nop_file.readlines()]