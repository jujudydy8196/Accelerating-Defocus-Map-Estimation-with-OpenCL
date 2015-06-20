#!/bin/bash

if [ "$1" == "-h" ]; then
	echo "Usage: `basename $0` [-h] <Input_pgm_dir>"
	exit 0
fi

cd ./src
make clean
make sparse

