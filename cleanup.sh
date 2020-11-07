#!/bin/bash

rm -f temp/train.*
rm -f temp/val.*
rm -f temp/test.*
rm -f temp/fin_tr.*
rm -f temp/*.data

echo "Cleaning /temp folder"

if [ "$1" == "--logs" ]; then
	rm -f results/exec_logs/*.results
	rm -f results/figures/*.png
	rm -rf tensorboard_logs/fit/
	echo "Cleaning /results/[exec_logs|figures] and tensorboard_logs/fit folders"
elif [ "$1" == "--models" ]; then
  rm -rf models_saved/*
  echo "Cleaning /models_saved folder"
elif [ "$1" == "--tuning" ]; then
  rm -rf tuning/*
  echo "Cleaning /tuning folder"
fi

echo "USAGE: $0 --[logs|models|tuning]"
