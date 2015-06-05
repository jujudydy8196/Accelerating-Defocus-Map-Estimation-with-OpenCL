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
    
    uchar* I_sparse = NULL;
    uchar* I_ori = NULL;
    uchar* I_gray = NULL;
    uchar* I_edge = NULL;
    int width, height, numPixel, r, mode;
    float lambda;

    sizePGM(width, height, argv[1]);
    lambda = atof(argv[4]);
    r = atof(argv[5]);
    mode = atoi(argv[6]);
    numPixel = width*height;
    int n = numPixel * 3;
    I_sparse = new uchar[numPixel];
    I_gray = new uchar[numPixel];
    I_edge = new uchar[numPixel];
    I_ori = new uchar[n];
    readPPM(I_ori, argv[1]);
    readPGM(I_gray, argv[2]);

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
