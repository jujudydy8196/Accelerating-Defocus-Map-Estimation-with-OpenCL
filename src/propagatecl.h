#ifndef PROPAGATELC_H
#define PROPAGATELC_H

#include <cstdlib>
#include <CL/cl.h>
#include "vec.h"

void loadKernels();
void propagatecl( const float*, const float*, const size_t, const size_t, const float, const size_t, Vec<float>& );
size_t getGlobalSize( int, size_t );
void printClMemory( int, cl_mem );

#endif