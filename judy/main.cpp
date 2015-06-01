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
	for(int w=0; w<width; w++) {
		for(int h=0; h<height; h++)
			cout << (int)I[w+h*width] << " ";
		cout << endl;
	}
   canny(I, height, width, 1.2, 0.5, 0.5, &edge, "test");

   writePGM(edge,width,height,"test.pgm");


}

