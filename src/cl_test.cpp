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

void test()
{
    device_manager->GetKernel("cl/test1.cl", "test1");
    device_manager->GetKernel("cl/test2.cl", "test2");
    device_manager->GetKernel("cl/test3.cl", "test3");
    device_manager->GetKernel("cl/test3.cl", "test4");
    device_manager->GetKernel("cl/test3.cl", "test4");

    int a[10] = {};
    const size_t block_dim[1] = { 100 };
    size_t grid_dim[1] = { 100 };

    cl_kernel kernel = device_manager->GetKernel("cl/test1.cl", "test1");
    auto d_out = device_manager->AllocateMemory(CL_MEM_READ_WRITE, 10*sizeof(int));
    vector<pair<const void*, size_t>> arg_and_sizes;
    arg_and_sizes.push_back( pair<const void*, size_t>( d_out.get(), sizeof(cl_mem) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, grid_dim, NULL, block_dim );

    LOG(INFO) << "after";
    for(size_t i = 0; i < 10; ++i){
        LOG(INFO) << a[i];
    }

    device_manager->ReadMemory(a, *d_out.get(), 10*sizeof(int));
    LOG(INFO) << "after";
    for(size_t i = 0; i < 10; ++i){
        LOG(INFO) << a[i];
    }

    kernel = device_manager->GetKernel("cl/test2.cl", "test2");
    device_manager->Call( kernel, arg_and_sizes, 1, grid_dim, NULL, block_dim );

    device_manager->ReadMemory(a, *d_out.get(), 10*sizeof(int));
    LOG(INFO) << "after";
    for(size_t i = 0; i < 10; ++i){
        LOG(INFO) << a[i];
    }

    kernel = device_manager->GetKernel("cl/test3.cl", "test3");
    device_manager->Call( kernel, arg_and_sizes, 1, grid_dim, NULL, block_dim );

    device_manager->ReadMemory(a, *d_out.get(), 10*sizeof(int));
    LOG(INFO) << "after";
    for(size_t i = 0; i < 10; ++i){
        LOG(INFO) << a[i];
    }

    kernel = device_manager->GetKernel("cl/test3.cl", "test4");
    device_manager->Call( kernel, arg_and_sizes, 1, grid_dim, NULL, block_dim );

    device_manager->ReadMemory(a, *d_out.get(), 10*sizeof(int));
    LOG(INFO) << "after";
    for(size_t i = 0; i < 10; ++i){
        LOG(INFO) << a[i];
    }
}

void testVec()
{
    // device_manager->GetKernel("vec_tmp.cl", "dot");
    device_manager->GetKernel("vec.cl", "vecDot");
    device_manager->GetKernel("vec.cl", "vecCopy");
    device_manager->GetKernel("vec.cl", "vecAdd");
    device_manager->GetKernel("vec.cl", "vecMultiply");
    device_manager->GetKernel("vec.cl", "vecDivide");
    device_manager->GetKernel("vec.cl", "vecScalorMultiply");
    device_manager->GetKernel("vec.cl", "vecTest");

    int size = 5;
    float a1 = 1, a2 = 1;
    float a[5] = { 1.0, 2.0, 4.0, 4.0, 5.0 };
    float b[5] = { 2.0, 2.5, 1.0, 4.0, 2.0 };
    float c[5] = {};
    float sum = 0;
    const size_t block_dim[1] = { 1024 };
    size_t grid_dim[1] = { 2048 };

    cl_kernel kernel = device_manager->GetKernel("vec.cl", "vecDot");
    auto d_v1 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_v2 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_result = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_sum = device_manager->AllocateMemory(CL_MEM_READ_WRITE, sizeof(float));
    device_manager->WriteMemory(a, *d_v1.get(), size*sizeof(float));
    device_manager->WriteMemory(b, *d_v2.get(), size*sizeof(float));

    vector<pair <const void*, size_t> > arg_and_sizes;
    arg_and_sizes.push_back( pair<const void*, size_t>( d_sum.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_v1.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_v2.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( NULL, block_dim[0]*sizeof(float) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, grid_dim, NULL, block_dim );

    device_manager->ReadMemory(&sum, *d_sum.get(), sizeof(float));
    LOG(INFO) << "dot";
    LOG(INFO) << sum;

    kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
    vector<pair <const void*, size_t> > arg_and_sizes_add;
    arg_and_sizes_add.push_back( pair<const void*, size_t>( d_result.get(), sizeof(cl_mem) ) );
    arg_and_sizes_add.push_back( pair<const void*, size_t>( d_v1.get(), sizeof(cl_mem) ) );
    arg_and_sizes_add.push_back( pair<const void*, size_t>( d_v2.get(), sizeof(cl_mem) ) );
    // arg_and_sizes_add.push_back( pair<const void*, size_t>( &a1, sizeof(float) ) );
    // arg_and_sizes_add.push_back( pair<const void*, size_t>( &a2, sizeof(float) ) );
    arg_and_sizes_add.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes_add, 1, grid_dim, NULL, block_dim );

    device_manager->ReadMemory(c, *d_result.get(), size*sizeof(float));
    LOG(INFO) << "add";
    for(size_t i = 0; i < size; ++i){
        LOG(INFO) << c[i];
    }

    kernel = device_manager->GetKernel("vec.cl", "vecTest");
    float test[128];
    int testSize = 128;
    auto d_test = device_manager->AllocateMemory(CL_MEM_READ_WRITE, 128*sizeof(float));
    arg_and_sizes.clear();
    arg_and_sizes.push_back( pair<const void*, size_t>( d_test.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &testSize, sizeof(int) ) );
    LOG(INFO) << arg_and_sizes.size();
    device_manager->Call( kernel, arg_and_sizes, 1, grid_dim, NULL, block_dim );
    device_manager->ReadMemory(test, *d_test.get(), testSize*sizeof(float));
    LOG(INFO) << "test";
    for(size_t i = 0; i < testSize; ++i){
        LOG(INFO) << i << " : " << test[i];
    }
}

int main( int argc, char** argv )
{
    InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    InitOpenCL(0);

    //test();
    testVec();

    return 0;
}
