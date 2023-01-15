#!/usr/bin/env python3
import os
import logging
from yapsy.PluginManager import PluginManager
from lib.perturbation_interface import IPerturbation

class PerturbationsManager:
  def __init__(self, perturbations_path):
    self.plugin_manager = PluginManager(
      directories_list=[perturbations_path],
      categories_filter={
        'Perturbations': IPerturbation
      },
      plugin_info_ext='perturbation')
    self.plugin_manager.collectPlugins()
    self.last_perturbation_index = -1 # None applied yet
    self.loaded_perturbations = []
    self.iterator = None
    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.DEBUG)

  def get_next_perturbation(self):
    if self.iterator:
      try:
        return next(self.iterator)
      except StopIteration:
        return None

  def get_all_perturbations(self):
    return self.plugin_manager.getAllPlugins()