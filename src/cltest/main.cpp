#include <iostream>
#include <fstream>
#include <cstdlib>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <ctime>
#include <CL/cl.h>
#include "fileIO.h"
#include "global.h"
#include "cl_helper.h"
#include "boxfilter.h"

using namespace std;
using namespace google;

void imageUchar2Float( uchar* uImage, float* fImage, const int size );
void imageFloat2Uchar( float* fImage, uchar* uImage, const int size );
void imageGray( float* image, float* gray, int size );
void InitOpenCL(size_t id);

DeviceManager *device_manager;

int main( int argc, char** argv )
{
    if(argc !=3) {
        cout << "Usage: test <original image> <radius>" << endl;
        return -1;
    }

    InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    InitOpenCL(0);

    clock_t start, stop;

    double times[4] = {};
    uchar* I_ori_uchar = NULL;
    uchar* I_out_uchar = NULL;
    float* I_ori = NULL;
    float* I_out = NULL;
    float* I_gray = NULL;
    int width, height, numPixel, r, mode;
    r = atoi( argv[2] );
    cout << argv[2] << ' ' << r << endl;

    sizePGM(width, height, argv[1]);
    numPixel = width*height;
    int n = numPixel * 3;
    I_gray = new float[numPixel];
    I_ori = new float[n];
    I_ori_uchar = new uchar[n];
    I_out_uchar = new uchar[numPixel];
    I_out = new float[numPixel];
    readPPM(I_ori_uchar, argv[1]);
    imageUchar2Float( I_ori_uchar, I_ori, n );
    imageGray( I_ori, I_gray, numPixel );

    loadKernels();

    start = clock();
    boxfilterCXX( I_gray, I_out, width, height, r );
    times[0] = double( clock() - start );
    imageFloat2Uchar( I_out, I_out_uchar, numPixel );
    writePGM(I_out_uchar, width, height, "CXX.pgm");

    start = clock();
    boxfilterIntegralCXX( I_gray, I_out, width, height, r );
    times[1] = double( clock() - start );
    imageFloat2Uchar( I_out, I_out_uchar, numPixel );
    writePGM(I_out_uchar, width, height, "integralCXX.pgm");

    start = clock();
    boxfilterOCL( I_gray, I_out, width, height, r );
    times[2] = double( clock() - start );
    imageFloat2Uchar( I_out, I_out_uchar, numPixel );
    writePGM(I_out_uchar, width, height, "OCL.pgm");

    start = clock();
    boxfilterIntegralOCL( I_gray, I_out, width, height, r );
    times[3] = double( clock() - start );
    imageFloat2Uchar( I_out, I_out_uchar, numPixel );
    writePGM(I_out_uchar, width, height, "integralOCL.pgm");

    for (int i = 0; i < 4; ++i)
    {
        cout << times[i] << endl;
    }

    delete [] I_out;
    delete [] I_gray;
    delete [] I_ori;
    delete [] I_out_uchar;
    delete [] I_ori_uchar;
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