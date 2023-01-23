#!/usr/bin/env python3
from abc import ABC, abstractclassmethod
from yapsy.IPlugin import IPlugin
from ext_tools.DexWave.lib.obfuscation import Obfuscation
import logging

class IPerturbation(ABC, IPlugin):
  def __init__(self):
    super().__init__()
    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.DEBUG) 

  @abstractclassmethod
  def perturbate(self, obuscation: Obfuscation):
    raise NotImplementedError()   