#!/bin/bash

if [ "$1" == "--logs" ]; then
	rm -f results/exec_logs/*.results
	rm -f results/figures/*.png
	rm -rf results/images/*
	rm -rf results/tensorboard/fit/
	echo "Cleaning /results/{exec_logs|figures|tensorboard|images} folders"
elif [ "$1" == "--models" ]; then
  rm -rf saved_models/*
  echo "Cleaning /saved_models folder"
elif [ "$1" == "--tuning" ]; then
  rm -rf tuning/*
  echo "Cleaning /tuning folder"
elif [ "$1" == "--complete" ]; then
  ./cleanup.sh --logs --ignore
  ./cleanup.sh --models --ignore
  ./cleanup.sh --tuning --ignore
fi

if [ "$2" != "--ignore" ]; then
  rm -f temp/train.*
  rm -f temp/val.*
  rm -f temp/test.*
  rm -f temp/fin_tr.*
  rm -f temp/*.data
  echo "Cleaning /temp folder"
  echo "USAGE: $0 --[logs|models|tuning|complete]"
fi
