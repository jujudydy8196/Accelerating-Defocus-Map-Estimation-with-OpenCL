#pragma once
#include <CL/cl.h>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <utility>
using namespace std;

const char* clewErrorString(cl_int error);
vector<cl_platform_id> GetPlatforms();
vector<char> GetPlatformName(const cl_platform_id pid);
vector<cl_device_id> GetPlatformDevices(const cl_platform_id pid);
vector<char> GetDeviceName(const cl_device_id did);
vector<cl_ulong> GetGlobalMemSize(const cl_device_id did);
vector<cl_ulong> GetGlobalMemCacheSize(const cl_device_id did);
vector<cl_ulong> GetLocalMemSize(const cl_device_id did);
vector<size_t> GetGroupSize(const cl_device_id did);
vector<cl_uint> GetWorkItemDim(const cl_device_id did);
vector<size_t> GetWorkItemSize(const cl_device_id did);
vector<cl_uint> GetComputeUnits(const cl_device_id did);
vector<char> GetKernelName(const cl_kernel kernel);

typedef unique_ptr<cl_mem, void(*)(cl_mem*)> MemoryObject;

// A simple class that wrap one device
class DeviceManager {
	cl_device_id device_;
	cl_context context_;
	cl_command_queue command_queue_;
	struct program_and_kernels {
		cl_program program_;
		map<string, cl_kernel> kernels_;
	};
	map<string, program_and_kernels> programs_;
public:
	DeviceManager(const cl_device_id device);
	virtual ~DeviceManager();

	// program_name = file name
	cl_kernel GetKernel(const string &program_name, const string &kernel_name);
	MemoryObject AllocateMemory(cl_mem_flags flags, size_t nbyte);
	void ReadMemory(void *host, cl_mem device, size_t nbyte);
	void WriteMemory(const void *host, cl_mem device, size_t nbyte);
	void Call(
		cl_kernel kernel,
		const vector<pair<const void*, size_t>> &arg_and_sizes,
		const cl_uint dim, const size_t *global_dim, const size_t *offset, const size_t *local_dim
	);
	void Finish();
};
