#ifndef BOXFILTER_H
#define BOXFILTER_H

#include <vector>
#include <iostream>
using namespace std;

void boxfilterCXX(const float* I, float* Out, int width, int height, int r);
void boxfilterIntegralCXX(const float* I, float* Out, int width, int height, int r);
void boxfilterOCL(const float* I, float* Out, int width, int height, int r);
void boxfilterIntegralOCL(const float* I, float* Out, int width, int height, int r);
void loadKernels();
size_t getGlobalSize( int size, size_t local_size );

#endif