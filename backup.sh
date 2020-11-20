#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M)

if [ "$(ls results/exec_logs/)" ]; then
  mkdir -p "BACKUP/${TIMESTAMP}"
  cp -r results/exec_logs/ BACKUP/${TIMESTAMP}/
  rm -f BACKUP/${TIMESTAMP}/exec_logs/.gitignore
fi

if [ "$(ls results/figures/)" ]; then
  mkdir -p "BACKUP/${TIMESTAMP}"
  cp -r results/figures/ BACKUP/${TIMESTAMP}/
  rm -f BACKUP/${TIMESTAMP}/figures/.gitignore
fi

if [ "$(ls results/tensorboard/)" ]; then
  mkdir -p "BACKUP/${TIMESTAMP}"
  cp -r results/tensorboard/ BACKUP/${TIMESTAMP}/
  rm -f BACKUP/${TIMESTAMP}/tensorboard/.gitignore
fi

if [ "$(ls tuning/)" ]; then
  mkdir -p "BACKUP/${TIMESTAMP}"
  cp -r tuning/ BACKUP/${TIMESTAMP}/
  rm -f tuning/.gitignore
fi

if [ "$(ls models_saved/)" ]; then
  mkdir -p "BACKUP/${TIMESTAMP}"
  cp -r models_saved/ BACKUP/${TIMESTAMP}/
  rm -f models_saved/.gitignore
fi

echo "If not empty, copied results/exec_logs/, results/figures/, results/tensorboard/, tuning/ and models_saved/ to BACKUP/${TIMESTAMP}"
