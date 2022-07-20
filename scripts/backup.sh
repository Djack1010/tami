#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/.."

if [ "$#" -ne 1 ]; then
    echo "ERROR: Provide name for folder to store in /BACKUP"
    echo "USAGE: $0 NAME_FOLDER_TO_CREATE"
    exit
fi

#TIMESTAMP=$(date +%Y%m%d_%H%M)
TIMESTAMP=$1

if [ "$(ls ${SCRIPTPATH}/results/exec_logs/)" ]; then
  mkdir -p "${SCRIPTPATH}/BACKUP/${TIMESTAMP}"
  cp -r ${SCRIPTPATH}/results/exec_logs/ ${SCRIPTPATH}/BACKUP/${TIMESTAMP}/
  rm -f ${SCRIPTPATH}/BACKUP/${TIMESTAMP}/exec_logs/.gitignore
fi

if [ "$(ls ${SCRIPTPATH}/results/figures/)" ]; then
  mkdir -p "${SCRIPTPATH}/BACKUP/${TIMESTAMP}"
  cp -r ${SCRIPTPATH}/results/figures/ ${SCRIPTPATH}/BACKUP/${TIMESTAMP}/
  rm -f ${SCRIPTPATH}/BACKUP/${TIMESTAMP}/figures/.gitignore
fi

if [ "$(ls ${SCRIPTPATH}/results/tensorboard/)" ]; then
  mkdir -p "${SCRIPTPATH}/BACKUP/${TIMESTAMP}"
  cp -r ${SCRIPTPATH}/results/tensorboard/ ${SCRIPTPATH}/BACKUP/${TIMESTAMP}/
  rm -f ${SCRIPTPATH}/BACKUP/${TIMESTAMP}/tensorboard/.gitignore
fi

if [ "$(ls ${SCRIPTPATH}/results/images/)" ]; then
  mkdir -p "${SCRIPTPATH}/BACKUP/${TIMESTAMP}"
  cp -r ${SCRIPTPATH}/results/images/ ${SCRIPTPATH}/BACKUP/${TIMESTAMP}/
  rm -f ${SCRIPTPATH}/BACKUP/${TIMESTAMP}/images/.gitignore
fi

if [ "$(ls ${SCRIPTPATH}/tuning/)" ]; then
  mkdir -p "${SCRIPTPATH}/BACKUP/${TIMESTAMP}"
  cp -r ${SCRIPTPATH}/tuning/ ${SCRIPTPATH}/BACKUP/${TIMESTAMP}/
  rm -f ${SCRIPTPATH}/tuning/.gitignore
fi

if [ "$(ls saved_models/)" ]; then
  mkdir -p "${SCRIPTPATH}/BACKUP/${TIMESTAMP}"
  cp -r ${SCRIPTPATH}/saved_models/ ${SCRIPTPATH}/BACKUP/${TIMESTAMP}/
  rm -f ${SCRIPTPATH}/saved_models/.gitignore
fi

if [ -d "${SCRIPTPATH}/BACKUP/${TIMESTAMP}" ]; then
  echo "BACKUP CREATED on the $(date +%d-%m-%Y) at $(date +%H:%M)" >> ${SCRIPTPATH}/BACKUP/${TIMESTAMP}/info.txt
fi

echo "If not empty, copied results/exec_logs/, results/figures/, results/tensorboard/, tuning/ and models_saved/ to ${SCRIPTPATH}/BACKUP/${TIMESTAMP}"
