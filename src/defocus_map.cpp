#include <iostream>
#include <cstdlib>
#include "fileIO.h"
#include "edge.h"
#include "defocus.h"
#include "propagate.h"

using namespace std;

int main(int argc, char** argv) {
    
    if(argc !=7) {
        cout << "Usage: defocus_map <original image> <gray image> <full depth image> <lambda> <radius> <gradient_descent[1] / filtering[2]>" << endl;
        return -1;
    }

    uchar* I_ori_uchar = NULL;
    float* I_sparse = NULL;
    float* I_ori = NULL;
    float* I_gray = NULL;
    float* I_edge = NULL;
    int width, height, numPixel, r, mode;
    float lambda;

    sizePGM(width, height, argv[1]);
    lambda = atof(argv[4]);
    r = atof(argv[5]);
    mode = atoi(argv[6]);
    numPixel = width*height;
    int n = numPixel * 3;
    I_sparse = new float[numPixel];
    I_gray = new float[numPixel];
    I_edge = new float[numPixel];
    I_ori = new float[n];
    I_ori_uchar = new uchar[n];
    readPPM(I_ori_uchar, argv[1]);
    imageUchar2Float( I_ori_uchar, I_ori, n );
    imageGray( I_ori, I_gray, numPixel );
    // readPGM(I_gray, argv[2]);

    // find sparse defocus map
    canny(I_gray, height, width, 1.2, 0.5, 0.8, &I_edge, "test");
    defocusEstimation(I_gray, I_edge, I_sparse, 1.0, 0.001, 3, width, height) ;

    Vec<uchar> result( numPixel );
    if(mode==1) propagate( I_ori, I_sparse, width, height, lambda, r, result );
    else if(mode==2) propagate2( I_ori, I_sparse, width, height, lambda, r, result );
    else ;
    writePGM(result.getPtr(), width, height, "check_result.pgm");
    
    delete [] I_ori;
    delete [] I_gray;
    delete [] I_edge;
    delete [] I_sparse;
    return 0;
}

void imageUchar2Float( uchar* uImage, float* fImage, const int size )
{
    for( size_t i = 0; i < size; ++i ){
        fImage[i] = uImage[i] / 255.0f;
    }
}

void imageGray( float* image, float* gray, int size ){
    for( size_t i = 0; i < size; ++i ){
        gray[i] = 0.2126*image[3*i] + 0.7152*image[3*i+1] + 0.0722*image[3*i+2];
    }
}