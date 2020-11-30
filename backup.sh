#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "ERROR: Provide name for folder to store in /BACKUP"
    echo "USAGE: $0 NAME_FOLDER_TO_CREATE"
    exit
fi

#TIMESTAMP=$(date +%Y%m%d_%H%M)
TIMESTAMP=$1

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

if [ "$(ls results/images/)" ]; then
  mkdir -p "BACKUP/${TIMESTAMP}"
  cp -r results/images/ BACKUP/${TIMESTAMP}/
  rm -f BACKUP/${TIMESTAMP}/images/.gitignore
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

if [ -d "BACKUP/${TIMESTAMP}" ]; then
  echo "BACKUP CREATED on the $(date +%d-%m-%Y) at $(date +%H:%M)" >> BACKUP/${TIMESTAMP}/info.txt
fi

echo "If not empty, copied results/exec_logs/, results/figures/, results/tensorboard/, tuning/ and models_saved/ to BACKUP/${TIMESTAMP}"
