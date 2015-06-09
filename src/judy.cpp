#include <iostream>
#include "fileIO.h"
#include "edge.h"
#include "defocus.h"

using namespace std;

typedef unsigned char uchar;


int main() {
	int width, height;
	sizePGM(width,height,"/Users/judy/Documents/senior/3DMM/final/judy/input.pgm");

	uchar* I = new uchar[width*height];
	uchar* edge = new uchar[width*height];
	readPGM(I,"/Users/judy/Documents/senior/3DMM/final/judy/input.pgm");
  canny(I, height, width, 1.2, 0.5, 0.8, &edge, "test");
  writePGM(edge,width,height,"test.pgm");

   uchar* sparseDMap= defocusEstimation(I, edge, 1.0, 0.001, 3, width, height) ;

}

