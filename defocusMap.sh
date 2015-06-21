#!/bin/bash

if [ "$1" = "-h" ]; then
	echo "Usage: sh `basename $0` [-h] <Input_pgm_dir_name> <propagte:gradient_descent[1] / filtering[2] / opencl[3] / compare [4]>
    <Input_pgm_name>: input image directory place in ./ImageData/ 
    example:
           --- src/
           |__ ImageData/
              |__ boy
                 |__ input.ppm
    ->  sh defocusMap.sh boy 3"
	exit 0
fi

cd ./src
rm time.txt
#make clean
#make sparse

#Sparse map
echo "sparse Map $1"
./sparse ../ImageData/$1/input.ppm 2

#make defocus_map
echo -n "propagate defocus sparse map "
case "$2" in
		1) echo "with gradient_descent"
				;;
		2) echo "with filtereing"
				;;
		3) echo "with opencl accelation"
				;;
		4) echo "comparing c++ with opencl"
				;;
esac

if [ "$2" = "4" ]; then
		./defocus_map ../ImageData/$1/input.ppm sparse.pgm 2 9 1
		./defocus_map ../ImageData/$1/input.ppm sparse.pgm 2 9 3
#defocus_map <original image> <sparse image> <lambda> <radius> <gradient_descent[1] / filtering[2] / opencl[3]>
else
		./defocus_map ../ImageData/$1/input.ppm sparse.pgm 2 9 $2
fi

echo "blur map result"
./blurmap/testimg ../ImageData/$1/input.ppm check_result.pgm


mv sparse.pgm ../ImageData/$1/
mv check_result.pgm ../ImageData/$1/
mv time.txt ../ImageData/$1/
mv ./blurredImage.pgm ../ImageData/$1/

