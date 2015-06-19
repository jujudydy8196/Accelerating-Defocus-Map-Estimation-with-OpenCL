#ifndef PROPAGATE_H
#define PROPAGATE_H

#include "vec.h"
#include <iostream>
#include <fstream>
using namespace std;

void propagate( const float*, const float*, const size_t, const size_t, const float, const size_t, Vec<float>& );
void constructH( const float*, Vec<float>& H, const size_t);
void constructEstimate( const float*, Vec<float>& );
void conjgrad( const Vec<float>&, float*,const LaplaMat*, float*, const Vec<float>&, Vec<float>&, size_t, size_t, float );
void vecFloat2uchar( const Vec<float>&, Vec<uchar>& );

void propagate2( float*, float*, size_t, size_t, float, size_t, Vec<float>& );
void vecUchar2float( const Vec<uchar>&, Vec<float>& );
void constructHE( const float*, Vec<float>&, Vec<float>& );
void constructHE( const Vec<float>&, Vec<float>&, Vec<float>& );
void checkHE( const Vec<float>&, const Vec<float>& );

void HFilter(float* , const float* , const Vec<float>& , const size_t);
void lambda_LFilter(float*, const float*, const float*, const int, const int, const float, const int );
void getAp(float*, const float*, const float*, const int);
void printEstimate( const Vec<float>& , const size_t, ofstream&);
void printP( const Vec<float>& , const size_t, ofstream&); 



#endif
