#include <iostream>
#include <math.h>

#define pi 3.1415
using namespace std;
typedef unsigned char uchar;

uchar* defocusEstimation(uchar* I, uchar* edge, float std, float lamda, float maxBlur, int width, int height);
void g1x(float* g, int* x, int* y, float std, int w);
void g1y(float* g, int* x, int* y, float std, int w);
void filter(float* gim, float* g , uchar* I, int width, int height, int w);

