#include <iostream>
#include "fileIO.h"
#include "edge.h"
#include "defocus.h"

using namespace std;

typedef unsigned char uchar;

void imageGray( float* image, float* gray, int size );

void imageUchar2Float( uchar* uImage, float* fImage, const int size );


int main(int argc, char** argv) {
	if(argc !=2) {
	    cout << "Usage: sparse_map <original image> " << endl;
	    return -1;
	}

	uchar* I_ori_uchar = NULL;
    float* I_sparse = NULL;
    float* I_ori = NULL;
    float* I_gray = NULL;
    float* I_edge = NULL;
    int width, height, numPixel, n;

    sizePGM(width, height, argv[1]);
    numPixel = width*height;
    n = numPixel * 3;

    I_sparse = new float[numPixel];
    I_gray = new float[numPixel];
    I_edge = new float[numPixel];
    I_ori = new float[n];
    I_ori_uchar = new uchar[n];

    readPPM(I_ori_uchar, argv[1]);
    imageUchar2Float( I_ori_uchar, I_ori, n );

    imageGray( I_ori, I_gray, numPixel );

    canny(I_gray, height, width, 1.2, 0.5, 0.8, &I_edge, "test");
    defocusEstimation(I_gray, I_edge, I_sparse, 1.0, 0.001, 3, width, height) ;


}

void imageGray( float* image, float* gray, int size ){
    for( size_t i = 0; i < size; ++i ){
        gray[i] = 0.2126*image[3*i] + 0.7152*image[3*i+1] + 0.0722*image[3*i+2];
    }
}

void imageUchar2Float( uchar* uImage, float* fImage, const int size )
{
    for( size_t i = 0; i < size; ++i ){
        fImage[i] = uImage[i] / 255.0f;
    }
}
