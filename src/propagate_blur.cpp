#include <iostream>
#include <cstdlib>
#include "fileIO.h"
#include "propagate.h"

using namespace std;

int main(int argc, char** argv) {
	
	if(argc != 7) {
		cout << "Usage: propagate_blur <original image> <sparse depth image> <full depth image> <lambda> <radius> <gradient_descent[1] / filtering[2]>" << endl;
		return -1;
	}
	
	uchar* I_sparse;
	// uchar* I_full;
	uchar* I_ori;
	int width, height, numPixel, r, mode;
	float lambda;

	sizePGM(width, height, argv[1]);
	lambda = atof(argv[4]);
	r = atof(argv[5]);
	mode = atoi(argv[6]);
	numPixel = width*height;
	I_sparse = new uchar[numPixel];
	int n = numPixel * 3;
	I_ori = new uchar[n];
	readPGM(I_sparse, argv[2]);
	readPPM(I_ori, argv[1]);

	Vec<uchar> result( numPixel );
	if(mode==1) propagate( I_ori, I_sparse, width, height, lambda, r, result );
	else if(mode==2) propagate2( I_ori, I_sparse, width, height, lambda, r, result );
	else ;
	writePGM(result.getPtr(), width, height, "check_result.pgm");
	
	// system("PAUSE");
	return 0;
}
