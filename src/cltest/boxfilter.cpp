#include "boxfilter.h"
#include "global.h"
#include "cl_helper.h"

void boxfilterCXX(const float* I, float* Out, int width, int height, int r)
{
    // for(size_t i = 0; i < 100; ++i){
    for( int y = 0; y < height; ++y ){
        for( int x = 0; x < width; ++x ){
            float sum = 0, count = 0;
            for( int dy = -r; dy <= r; ++dy ){
                int yid = y + dy;
                if( yid < 0 || yid >= height ) continue;
                for( int dx = -r; dx <= r; ++dx ){
                    int xid = x + dx;
                    if( xid < 0 || xid >= width ) continue;
                    sum += I[yid*width+xid];
                    ++count;
                }
            }
            Out[y*width+x] = sum / count;
        }
    }
// }
}

void boxfilterIntegralCXX(const float* I, float* Out, int width, int height, int r)
{
    float* tmp = new float[width*height];
    
    // cumulative sum over Y axis
    // for(size_t i = 0; i < 100; ++i){
    for(int x = 0; x < width; ++x) {
        // y = 0
        tmp[x] = I[x];
        // y > 0
        for(int y = 1; y < height; ++y) {
            tmp[y*width+x] = tmp[(y-1)*width+x] + I[y*width+x];
        }
    }

    // difference over Y axis
    // imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
    for(int y = 0; y <= r; ++y) {
    for(int x = 0; x < width; ++x) {
        Out[y*width+x] = tmp[(y+r)*width+x];
    }
    }
    // imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);
    for(int y = r+1; y < height-r; ++y) {
    for(int x = 0; x < width; ++x) {
        Out[y*width+x] = tmp[(y+r)*width+x] - tmp[(y-r-1)*width+x];
    }
    }
    // imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);
    for(int y = height-r; y < height; ++y) {
    for(int x = 0; x < width; ++x) {
        Out[y*width+x] = tmp[(height-1)*width+x] - tmp[(y-r-1)*width+x];
    }
    }

    // cumulative sum over X axis
    for(int y = 0; y < height; ++y) {
        // x = 0
        tmp[y*width] = Out[y*width];
        // x > 0
        for(int x = 1; x < width; ++x) {
            tmp[y*width+x] = tmp[y*width+x-1] + Out[y*width+x];
        }
    }
    // difference over X axis
    // imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);
    for(int x = 0; x <= r; ++x) {
    for(int y = 0; y < height; ++y) {
        Out[y*width+x] = tmp[y*width+x+r];
    }
    }
    // imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);
    for(int x = r+1; x < width-r; ++x) {
    for(int y = 0; y < height; ++y) {
        Out[y*width+x] = tmp[y*width+x+r] - tmp[y*width+x-r-1];
    }
    }
    // imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);
    for(int x = width-r; x < width; ++x) {
    for(int y = 0; y < height; ++y) {
        Out[y*width+x] = tmp[y*width+width-1] - tmp[y*width+x-r-1];
    }
    }
// }
    delete [] tmp;
}

void boxfilterOCL(const float* I, float* Out, int width, int height, int r)
{   
    int size = width * height;

    auto d_I = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_out = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    device_manager->WriteMemory( I, *d_I.get(), size*sizeof(float));

    cl_kernel kernel = device_manager->GetKernel("boxfilter.cl", "boxfilter");
    size_t local_size[2] = {32,32};
    size_t global_size[2] = {};
    vector<pair<const void*, size_t>> arg_and_sizes;
    global_size[0] = getGlobalSize( width, local_size[0] );
    global_size[1] = getGlobalSize( height, local_size[1] );

    cout << global_size[0] << ' ' << global_size[1] << endl;

    arg_and_sizes.push_back( pair<const void*, size_t>( d_out.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_I.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &r, sizeof(int) ) );
    // for(size_t i = 0; i < 100; ++i)
    device_manager->Call( kernel, arg_and_sizes, 2, global_size, NULL, local_size );

    device_manager->ReadMemory(Out, *d_out.get(), size*sizeof(float));
}

void boxfilterIntegralOCL(const float* I, float* Out, int width, int height, int r)
{
    int size = width * height;

    auto d_I = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_out = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_box_tmp = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_box_buffer = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    device_manager->WriteMemory( I, *d_I.get(), size*sizeof(float));

    cl_kernel kernel = device_manager->GetKernel("boxfilter.cl", "boxfilterCumulateY");
    size_t local_size[1] = {1024};
    size_t global_size[1] = {};
    size_t local16[] = {16};
    vector<pair<const void*, size_t>> arg_and_sizes;
    global_size[0] = getGlobalSize( size, local_size[0] );

    arg_and_sizes.push_back( pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_I.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_box_buffer.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &r, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, local_size, NULL, local_size );

    kernel = device_manager->GetKernel("boxfilter.cl", "boxfilterCumulateX");
    arg_and_sizes[0] = pair<const void*, size_t>( d_out.get(), sizeof(cl_mem) );
    arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
    device_manager->Call( kernel, arg_and_sizes, 1, local_size, NULL, local16 );

    device_manager->ReadMemory(Out, *d_out.get(), size*sizeof(float));
}

void loadKernels()
{
    device_manager->GetKernel("boxfilter.cl", "boxfilter");
    device_manager->GetKernel("boxfilter.cl", "boxfilterCumulateX");
    device_manager->GetKernel("boxfilter.cl", "boxfilterCumulateY");
}

size_t getGlobalSize( int size, size_t local_size )
{
    if( size % local_size )
        return local_size * ( size / local_size + 1 );
    else return size;
}