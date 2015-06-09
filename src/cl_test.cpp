#include <glog/logging.h>
#include <gflags/gflags.h>
#include <CL/cl.h>
#include "cl_helper.h"
using namespace google;

DeviceManager *device_manager;

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

int main( int argc, char** argv )
{
    InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    InitOpenCL(0);


    return 0;
}