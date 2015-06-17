#include "vec.h"

LaplaMat::LaplaMat(const float* I_ori, const size_t width, const size_t height, const size_t r):_r(r), _width(width), _height(height) {
    _gf = new guided_filter(I_ori, width, height, r, 0.00001);
    int numPixel = _width * _height;
    _I_ori = new float[numPixel];
    for(int i = 0; i < numPixel; ++i) {
        _I_ori[i] = I_ori[i];
    }
}

void LaplaMat::run(float* Lp, const float* p, const float lambda) const {
    int numWinPixel, numPixel;
    numWinPixel = _r*_r;
    numPixel = _width * _height;
    float* tmpI = new float[numPixel];
    _gf->run(p, tmpI);
    
    for(int i = 0; i < numPixel; ++i) {
            //I-W*I
            //tmpI[i] = (float)_I_ori[i] - tmpI[i];
            // of1<<tmpI[i]<<" ";
            tmpI[i] = p[i] - tmpI[i];
            // of2<<tmpI[i]<<" ";
            //L = |w|*(I-W)     //L = lamda*L
            Lp[i] = (float)lambda * (float)numWinPixel * tmpI[i];
            //of2<<Lp[i]<<" ";
    }

    delete [] tmpI;
}

LaplaMat::~LaplaMat() {
    delete _gf;
}