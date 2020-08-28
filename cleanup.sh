#!/bin/bash

rm -f temp/train.*
rm -f temp/val.*
rm -f temp/test.*
rm -f temp/fin_tr.*
rm -f temp/*.data

echo "Cleaning /temp folder"

if [ "$1" == "--complete" ]; then
	rm -f preprocessed_dataset/*.data
	rm -f results/*.results
	echo "Cleaning /preprocesed_dataset and /results folders"
fi

echo "USAGE: $0 [--complete]"
