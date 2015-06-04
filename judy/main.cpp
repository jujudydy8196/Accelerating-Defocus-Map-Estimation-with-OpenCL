#include <iostream>
#include "fileIO.h"
#include "edge.h"
#include "defocus.h"

using namespace std;

typedef unsigned char uchar;


int main(int argc, char** argv) {
	int width, height;
	sizePGM(width,height,argv[1]);

	uchar* I = new uchar[width*height];
	uchar* edge = new uchar[width*height];
	readPGM(I,"input.pgm");
	// for(int h=0; h<height; h++) {
	// 	for(int w=0; w<width; w++)
	// 		cout << (float)I[w+h*height]/255.0 << " ";
	// 	cout << endl;
	// }

   canny(I, height, width, 1.2, 0.5, 0.8, &edge, "test");
   writePGM(edge,width,height,"test.pgm");

	// for(int h=0; h<height; h++) {
	// 	for(int w=0; w<width; w++) {
	// 		I[w+h*height] = (float)I[w+h*height]/255.0;
	// 		cout << (float)I[w+h*height] << " ";
	// 	}
	// 	cout << endl;
	// }


   uchar* sparseDMap= defocusEstimation(I, edge, 1.0, 0.001, 3, width, height) ;
   writePGM(sparseDMap,width,height,"sparse.pgm");



}

