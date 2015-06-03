#include <iostream>
#include <cstdlib>
#include "fileIO.h"
#include "propagate.h"

using namespace std;

int main(int argc, char** argv) {
	
	if(argc != 6) {
		cout << "Usage: propagate_blur <original image> <sparse depth image> <full depth image> <lambda> <radius>" << endl;
		return -1;
	}
	
	uchar* I_sparse;
	// uchar* I_full;
	uchar* I_ori;
	int width, height, numPixel, r;
	float lambda;

	sizePGM(width, height, argv[1]);
	lambda = atof(argv[4]);
	r = atof(argv[5]);
	numPixel = width*height;
	I_sparse = new uchar[numPixel];
	int n = numPixel * 3;
	I_ori = new uchar[n];
	readPGM(I_sparse, argv[2]);
	readPPM(I_ori, argv[1]);

	Vec<uchar> result( numPixel );
	propagate2( I_ori, I_sparse, width, height, lambda, r, result );
	// propagate( I_ori, I_sparse, width, height, lambda, r, result );

	writePGM(result.getPtr(), width, height, "check_result.pgm");
	
	// system("PAUSE");
	return 0;
}