#include <iostream>
#include <math.h>

using namespace std;
typedef unsigned char uchar;

uchar* defocusEstimation(uchar* I, uchar* edge, float std, float lamda, float maxBlur);
void g1x(float* g, int* x, int* y, float std, int w);
void g1y(float* g, int* x, int* y, float std, int w);

