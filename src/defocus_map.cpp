#include <iostream>
#include <fstream>
#include <cstdlib>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <ctime>
#include <CL/cl.h>
#include "fileIO.h"
#include "edge.h"
#include "defocus.h"
#include "propagate.h"
#include "propagatecl.h"
#include "global.h"
#include "cl_helper.h"

using namespace std;
using namespace google;

void imageUchar2Float( uchar* uImage, float* fImage, const int size );
void imageFloat2Uchar( float* fImage, uchar* uImage, const int size );
void imageGray( float* image, float* gray, int size );
void InitOpenCL(size_t id);

DeviceManager *device_manager;

int main(int argc, char** argv) {
    clock_t start, stop;
    
    if(argc !=5) {
        cout << "Usage: defocus_map <original image> <lambda> <radius> <gradient_descent[1] / filtering[2] / opencl[3]>" << endl;
        return -1;
    }

    InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    InitOpenCL(0);

    uchar* I_ori_uchar = NULL;
    uchar* I_ori2_uchar = NULL;
    uchar* I_sparse_uchar = NULL;
    uchar* I_out = NULL;
    float* I_sparse = NULL;
    float* I_sparse2 = NULL;
    float* I_ori = NULL;
    float* I_gray = NULL;
    float* I_edge = NULL;
    int width, height, numPixel, r, mode;
    float lambda;

    sizePGM(width, height, argv[1]);
    lambda = atof(argv[2]);
    r = atof(argv[3]);
    mode = atoi(argv[4]);
    numPixel = width*height;
    int n = numPixel * 3;
    I_sparse = new float[numPixel];
    I_sparse2 = new float[numPixel];
    I_gray = new float[numPixel];
    I_edge = new float[numPixel];
    I_ori = new float[n];
    I_ori_uchar = new uchar[n];
    I_ori2_uchar = new uchar[n];
    I_sparse_uchar = new uchar[numPixel];
    I_out = new uchar[numPixel];
    readPPM(I_ori_uchar, argv[1]);
    imageUchar2Float( I_ori_uchar, I_ori, n );

    readPGM(I_sparse_uchar, "./sparse.pgm"); 
    imageUchar2Float( I_sparse_uchar, I_sparse2, numPixel );
    // imageFloat2Uchar( I_sparse, I_out, numPixel );

    imageGray( I_ori, I_gray, numPixel );
    // readPGM(I_gray, argv[2]);

    // find sparse defocus map
    // canny(I_gray, height, width, 1.2, 0.5, 0.8, &I_edge, "test");
    // writePGM(I_edge,width,height,"test.pgm");

    // defocusEstimation(I_gray, I_edge, I_sparse, 1.0, 0.001, 3, width, height) ;

    // float tmp;
    // for(  size_t i = 0; i < numPixel; ++i){
    //     tmp = (I_sparse[i] - I_sparse2[i]) * 255;
    //     if(tmp) cout << tmp << " ";
    // }

    imageFloat2Uchar(I_ori, I_ori2_uchar, n);
    writePPM(I_ori2_uchar, width, height,"color.ppm");
    imageFloat2Uchar(I_sparse2, I_ori2_uchar, numPixel);
    writePGM(I_ori2_uchar, width, height,"sparse2.pgm");
    Vec<float> result( numPixel );
    start = clock();
    if(mode==1) propagate( I_ori, I_sparse2, width, height, lambda, r, result );
    else if(mode==2) propagate2( I_ori, I_sparse, width, height, lambda, r, result );
    else if(mode==3) propagatecl( I_ori, I_sparse2, width, height, lambda, r, result );
    stop = clock();

    cout << "propagate time: " << double(stop - start) / CLOCKS_PER_SEC <<endl; 
    
    ofstream timelog;
    timelog.open("time.txt", ios::app);
    if (mode==1) 
        timelog << "propagate time: " ;
    else if (mode==3)
        timelog << "cl_propagate time: " ;
    timelog << << double(stop - start) / CLOCKS_PER_SEC <<endl;
    timelog.close();

    imageFloat2Uchar( result.getPtr(), I_out, numPixel );
    writePGM(I_out, width, height, "check_result.pgm");
    
    delete [] I_ori;
    delete [] I_gray;
    delete [] I_edge;
    delete [] I_sparse;
    delete [] I_sparse2;
    delete [] I_ori2_uchar;
    delete [] I_ori_uchar;
    delete [] I_sparse_uchar;
    delete [] I_out;
    return 0;
}

void imageUchar2Float( uchar* uImage, float* fImage, const int size )
{
    for( size_t i = 0; i < size; ++i ){
        fImage[i] = uImage[i] / 255.0f;
    }
}

void imageFloat2Uchar( float* fImage, uchar* uImage, const int size )
{
    for( size_t i = 0; i < size; ++i ){
        uImage[i] = uchar(fImage[i] * 255.0);
    }
}

void imageGray( float* image, float* gray, int size ){
    for( size_t i = 0; i < size; ++i ){
        gray[i] = 0.2126*image[3*i] + 0.7152*image[3*i+1] + 0.0722*image[3*i+2];
    }
}

void InitOpenCL(size_t id)
{
    // platforms
    auto platforms = GetPlatforms();
    LOG(INFO) << platforms.size() << " platform(s) found";
    int last_nvidia_platform = -1;
    for (size_t i = 0; i < platforms.size(); ++i) {
        auto platform_name = GetPlatformName(platforms[i]);
        LOG(INFO) << ">>> Name: " << platform_name.data();
        if (strcmp("NVIDIA CUDA", platform_name.data()) == 0) {
            last_nvidia_platform = i;
        }
    }
    CHECK_NE(last_nvidia_platform, -1) << "Cannot find any NVIDIA CUDA platform";

    // devices under the last CUDA platform
    auto devices = GetPlatformDevices(platforms[last_nvidia_platform]);
    LOG(INFO) << devices.size() << " device(s) found under some platform";
    for (size_t i = 0; i < devices.size(); ++i) {
        auto device_name = GetDeviceName(devices[i]);
        LOG(INFO) << ">>> Name: " << device_name.data();
    }
    CHECK_LT(id, devices.size()) << "Cannot find device " << id;

    auto global_size = GetGlobalMemSize( devices[id] );
    LOG(INFO) << "global mem size : " << global_size[0];
    auto global_cache_casize = GetGlobalMemCacheSize( devices[id] );
    LOG(INFO) << "global mem cache size : " << global_cache_casize[0];
    auto local_size = GetLocalMemSize( devices[id] );
    LOG(INFO) << "local mem size : " << local_size[0];
    auto groupSize = GetGroupSize( devices[id] );
    LOG(INFO) << "group size : " << groupSize[0];
    auto workDim = GetWorkItemDim( devices[id] );
    LOG(INFO) << "work dim : " << workDim[0];
    auto workSize = GetWorkItemSize( devices[id] );
    LOG(INFO) << "work size : " << workSize[0] << ' ' << workSize[1] << ' ' << workSize[2];
    auto computeUnits = GetComputeUnits( devices[id] );
    LOG(INFO) << "compute units : " << computeUnits[0];

    device_manager = new DeviceManager(devices[id]);
}
