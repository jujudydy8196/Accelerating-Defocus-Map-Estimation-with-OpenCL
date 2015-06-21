#!/bin/bash

if [ "$1" = "-h" ]; then
	echo "Usage: `basename $0` [-h] <Input_pgm_dir_name>
    <Input_pgm_name>: input image directory place in ./ImageData/ 
    example:
           --- src/
           |__ ImageData/
              |__ boy
                 |__ input.ppm
    ->  ./defocusMap.sh boy "
	exit 0
fi

cd ./src
make clean
make sparse

