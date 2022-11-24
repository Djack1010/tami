#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/.."

if [ "$1" == "--logs" ]; then
	rm -f ${SCRIPTPATH}/results/exec_logs/*.results
	rm -f ${SCRIPTPATH}/results/figures/*.png
	rm -rf ${SCRIPTPATH}/results/images/*
	rm -rf ${SCRIPTPATH}/results/tensorboard/fit/
	echo "Cleaning /results/{exec_logs|figures|tensorboard|images} folders"
elif [ "$1" == "--models" ]; then
  rm -rf ${SCRIPTPATH}/saved_models/*
  echo "Cleaning /saved_models folder"
elif [ "$1" == "--tuning" ]; then
  rm -rf ${SCRIPTPATH}/tuning/*
  echo "Cleaning /tuning folder"
elif [ "$1" == "--complete" ]; then
  ${SCRIPTPATH}/scripts/cleanup.sh --logs --ignore
  ${SCRIPTPATH}/scripts/cleanup.sh --models --ignore
  ${SCRIPTPATH}/scripts/cleanup.sh --tuning --ignore
fi

if [ "$2" != "--ignore" ]; then
  rm -f ${SCRIPTPATH}/temp/train.*
  rm -f ${SCRIPTPATH}/temp/val.*
  rm -f ${SCRIPTPATH}/temp/test.*
  rm -f ${SCRIPTPATH}/temp/fin_tr.*
  rm -f ${SCRIPTPATH}/temp/*.data
  echo "Cleaning /temp folder"
  echo "USAGE: $0 --[logs|models|tuning|complete]"
fi
