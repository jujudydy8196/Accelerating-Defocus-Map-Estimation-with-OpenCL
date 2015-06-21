#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include "fileIO.h"

using namespace std;


void imageFloat2Uchar( float* fImage, uchar* uImage, const int size )
{
    for( size_t i = 0; i < size; ++i ){
        uImage[i] = uchar(fImage[i] * 255.0);
    }
}

int main(int argc, char** argv ) {
	int width, height, numPixel;
	uchar* blurmap;
	uchar* Im;
	uchar* back;
	uchar* front;
	sizePGM(width, height, argv[1]);
	cout<<"width "<<width<<" height "<<height;
	numPixel = width*height;
	blurmap = new uchar[numPixel];
	Im = new uchar[3*numPixel];
	back = new uchar[numPixel];
	front = new uchar[3*numPixel];
	readPPM(Im, argv[1]);
	readPGM(blurmap, argv[2]);
	int front_thresh = 145;


	for(int i = 0; i < numPixel; ++i) {
		if(blurmap[i] < front_thresh) {
			//background is white
			back[i] = 255;
			//frontground is colorful
			front[3*i  ] = Im[3*i  ];
			front[3*i+1] = Im[3*i+1];
			front[3*i+2] = Im[3*i+2];
		}
		else {
			//background is black
			back[i] = 0;
			//frontground is black
			front[3*i  ] = 0;
			front[3*i+1] = 0;
			front[3*i+2] = 0;	
		}
	}

	writePGM(back, width, height, "background.pgm");	
	writePPM(front, width, height, "frontground.ppm");
	delete [] blurmap;
	delete [] Im;
	delete [] back;
	delete [] front;
	return 0;

}
