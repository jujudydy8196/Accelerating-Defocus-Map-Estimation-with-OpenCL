#include "cl_helper.h"
#include <glog/logging.h>
#include <functional>
#include <cstdio>
using namespace std::placeholders;

// Copy from the Internet
const char* clewErrorString(cl_int error)
{
	static const char* strings[] =
	{
		"CL_SUCCESS"                        , "CL_DEVICE_NOT_FOUND"               ,
		"CL_DEVICE_NOT_AVAILABLE"           , "CL_COMPILER_NOT_AVAILABLE"         ,
		"CL_MEM_OBJECT_ALLOCATION_FAILURE"  , "CL_OUT_OF_RESOURCES"               ,
		"CL_OUT_OF_HOST_MEMORY"             , "CL_PROFILING_INFO_NOT_AVAILABLE"   ,
		"CL_MEM_COPY_OVERLAP"               , "CL_IMAGE_FORMAT_MISMATCH"          ,
		"CL_IMAGE_FORMAT_NOT_SUPPORTED"     , "CL_BUILD_PROGRAM_FAILURE"          ,
		"CL_MAP_FAILURE"                    , ""                                  ,
		""                                  , ""                                  ,
		""                                  , ""                                  ,
		""                                  , ""                                  ,
		""                                  , ""                                  ,
		""                                  , ""                                  ,
		""                                  , ""                                  ,
		""                                  , ""                                  ,
		""                                  , ""                                  ,
		"CL_INVALID_VALUE"                  , "CL_INVALID_DEVICE_TYPE"            ,
		"CL_INVALID_PLATFORM"               , "CL_INVALID_DEVICE"                 ,
		"CL_INVALID_CONTEXT"                , "CL_INVALID_QUEUE_PROPERTIES"       ,
		"CL_INVALID_COMMAND_QUEUE"          , "CL_INVALID_HOST_PTR"               ,
		"CL_INVALID_MEM_OBJECT"             , "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
		"CL_INVALID_IMAGE_SIZE"             , "CL_INVALID_SAMPLER"                ,
		"CL_INVALID_BINARY"                 , "CL_INVALID_BUILD_OPTIONS"          ,
		"CL_INVALID_PROGRAM"                , "CL_INVALID_PROGRAM_EXECUTABLE"     ,
		"CL_INVALID_KERNEL_NAME"            , "CL_INVALID_KERNEL_DEFINITION"      ,
		"CL_INVALID_KERNEL"                 , "CL_INVALID_ARG_INDEX"              ,
		"CL_INVALID_ARG_VALUE"              , "CL_INVALID_ARG_SIZE"               ,
		"CL_INVALID_KERNEL_ARGS"            , "CL_INVALID_WORK_DIMENSION"         ,
		"CL_INVALID_WORK_GROUP_SIZE"        , "CL_INVALID_WORK_ITEM_SIZE"         ,
		"CL_INVALID_GLOBAL_OFFSET"          , "CL_INVALID_EVENT_WAIT_LIST"        ,
		"CL_INVALID_EVENT"                  , "CL_INVALID_OPERATION"              ,
		"CL_INVALID_GL_OBJECT"              , "CL_INVALID_BUFFER_SIZE"            ,
		"CL_INVALID_MIP_LEVEL"              , "CL_INVALID_GLOBAL_WORK_SIZE"       ,
		"CL_UNKNOWN_ERROR_CODE"
	};

	if (error >= -63 && error <= 0)
		return strings[-error];
	else
		return strings[64];
}

template <class ReturnType, class SizeType, class GetFunctionType, class ... Parameters>
vector<ReturnType> OpenclAllocateRoutine(GetFunctionType get_function, Parameters ... args)
{
	cl_int result;

	// get num
	SizeType num;
	result = get_function(0, nullptr, &num, args...);
	CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
	CHECK_GT(num, 0);

	// get data
	vector<ReturnType> ret(num);
	SizeType num_got;
	result = get_function(num, ret.data(), &num_got, args...);
	CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
	CHECK_EQ(num, num_got);

	return ret;
}

vector<cl_platform_id> GetPlatforms()
{
	return OpenclAllocateRoutine<cl_platform_id, cl_uint>(clGetPlatformIDs);
}

vector<char> GetPlatformName(const cl_platform_id pid)
{
	return OpenclAllocateRoutine<char, size_t>(bind(clGetPlatformInfo, _4, (cl_platform_info)CL_PLATFORM_NAME, _1, _2, _3), pid);
}

vector<cl_device_id> GetPlatformDevices(const cl_platform_id pid)
{
	return OpenclAllocateRoutine<cl_device_id, cl_uint>(bind(clGetDeviceIDs, _4, (cl_device_type)CL_DEVICE_TYPE_GPU, _1, _2, _3), pid);
}

vector<char> GetDeviceName(const cl_device_id did)
{
	return OpenclAllocateRoutine<char, size_t>(bind(clGetDeviceInfo, _4, (cl_device_info)CL_DEVICE_NAME, _1, _2, _3), did);
}

vector<cl_ulong> GetGlobalMemSize(const cl_device_id did)
{
	return OpenclAllocateRoutine<cl_ulong, size_t>(bind(clGetDeviceInfo, _4, (cl_device_info)CL_DEVICE_GLOBAL_MEM_SIZE, _1, _2, _3), did);
}

vector<cl_ulong> GetGlobalMemCacheSize(const cl_device_id did)
{
	return OpenclAllocateRoutine<cl_ulong, size_t>(bind(clGetDeviceInfo, _4, (cl_device_info)CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, _1, _2, _3), did);
}

vector<cl_ulong> GetLocalMemSize(const cl_device_id did)
{
	return OpenclAllocateRoutine<cl_ulong, size_t>(bind(clGetDeviceInfo, _4, (cl_device_info)CL_DEVICE_LOCAL_MEM_SIZE, _1, _2, _3), did);
}

vector<size_t> GetGroupSize(const cl_device_id did)
{
	return OpenclAllocateRoutine<size_t, size_t>(bind(clGetDeviceInfo, _4, (cl_device_info)CL_DEVICE_MAX_WORK_GROUP_SIZE, _1, _2, _3), did);
}

vector<cl_uint> GetWorkItemDim(const cl_device_id did)
{
	return OpenclAllocateRoutine<cl_uint, size_t>(bind(clGetDeviceInfo, _4, (cl_device_info)CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, _1, _2, _3), did);
}

vector<size_t> GetWorkItemSize(const cl_device_id did)
{
	return OpenclAllocateRoutine<size_t, size_t>(bind(clGetDeviceInfo, _4, (cl_device_info)CL_DEVICE_MAX_WORK_ITEM_SIZES, _1, _2, _3), did);
}

vector<cl_uint> GetComputeUnits(const cl_device_id did)
{
	return OpenclAllocateRoutine<cl_uint, size_t>(bind(clGetDeviceInfo, _4, (cl_device_info)CL_DEVICE_MAX_COMPUTE_UNITS, _1, _2, _3), did);
}

vector<char> GetKernelName(const cl_kernel kernel)
{
	return OpenclAllocateRoutine<char, size_t>(bind(clGetKernelInfo, _4, (cl_kernel_info)CL_KERNEL_FUNCTION_NAME, _1, _2, _3), kernel);
}

DeviceManager::DeviceManager(const cl_device_id device): device_(device)
{
	cl_int result;
	context_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &result);
	CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
	command_queue_ = clCreateCommandQueue(context_, device_, 0, &result);
	CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
}

DeviceManager::~DeviceManager()
{
	cl_int result;
	for (auto &program_and_kernels: programs_) {
		for (auto &kernel: program_and_kernels.second.kernels_) {
			result = clReleaseKernel(kernel.second);
		}
		result = clReleaseProgram(program_and_kernels.second.program_);
	}
	result = clReleaseCommandQueue(command_queue_);
	CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
	result = clReleaseContext(context_);
	CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
}

unique_ptr<char[]> LoadFile(const char *file_name)
{
	FILE *fp = fopen(file_name, "rb");
	CHECK_NOTNULL(fp);

	// Get size
	fseek(fp, 0, SEEK_END);
	const long file_size = ftell(fp);
    DLOG(INFO) << file_name << " size: " << file_size << endl;

	// Now allocate and read
	unique_ptr<char[]> file_data(new char[file_size]);
	rewind(fp);
	fread(file_data.get(), 1, file_size, fp);
	fclose(fp);
    file_data[file_size-1] = '\0';

	return move(file_data);
}

cl_kernel DeviceManager::GetKernel(const string &program_name, const string &kernel_name)
{
	cl_int result;

	auto programs_it = programs_.find(program_name);
	if (programs_it == programs_.end()) {
		// program not exists
		DLOG(INFO) << "Program " << program_name << " not exists, create one";
		programs_it = programs_.insert(programs_it, make_pair(program_name, program_and_kernels()));
		program_and_kernels &program = programs_it->second;

		auto file_content = LoadFile(program_name.c_str());
		const char *content_pointer = file_content.get();
		program.program_ = clCreateProgramWithSource(
			context_, 1, &content_pointer, nullptr, &result
		);
		CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
		result = clBuildProgram(program.program_, 1, &device_, nullptr, nullptr, nullptr);
		// CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
		
		cl_build_status bresult;
		result = clGetProgramBuildInfo(
			program.program_, device_, CL_PROGRAM_BUILD_STATUS,
			sizeof(cl_build_status), &bresult, nullptr
		);
		CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);

		if (bresult != CL_BUILD_SUCCESS) {
			char compile_message[1024*64];
			result = clGetProgramBuildInfo(
				program.program_, device_, CL_PROGRAM_BUILD_LOG,
				sizeof(compile_message), compile_message, nullptr
			);
			CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
			LOG(FATAL) << compile_message;
		}
	}

	auto &program_and_kernels = programs_it->second;
	auto kernels_it = program_and_kernels.kernels_.find(kernel_name);
	if (kernels_it == program_and_kernels.kernels_.end()) {
		// kernel not exists
		DLOG(INFO) << "Kernel " << kernel_name << " not exists, create one";
		cl_kernel kernel = clCreateKernel(program_and_kernels.program_, kernel_name.c_str(), &result);
		CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
		program_and_kernels.kernels_.insert(kernels_it, make_pair(kernel_name, kernel));
		return kernel;
	} else {
		// kernel exists
		return kernels_it->second;
	}

	// to supress warning
	return cl_kernel();
}

auto cl_mem_deleter = [](cl_mem *m) -> void
{
	DLOG(INFO) << "cl_mem: " << *m << " freed";
	clReleaseMemObject(*m);
	delete m;
};

MemoryObject DeviceManager::AllocateMemory(cl_mem_flags flags, size_t nbyte)
{
	cl_int result;
	cl_mem memory_object = clCreateBuffer(context_, flags, nbyte, NULL, &result);
	CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
	return move(MemoryObject(new cl_mem(memory_object), cl_mem_deleter));
}

void DeviceManager::ReadMemory(void *host, cl_mem device, size_t nbyte)
{
	cl_int result = clEnqueueReadBuffer(
		command_queue_,
		device,
		CL_TRUE,
		0, nbyte,
		host,
		0, nullptr, nullptr
	);
	CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
}

void DeviceManager::WriteMemory(const void *host, cl_mem device, size_t nbyte)
{
	// TODO: a OpenCL call is required here
	cl_int result = clEnqueueWriteBuffer(
		command_queue_,
		device,
		CL_TRUE,
		0, nbyte,
		host,
		0, nullptr, nullptr
	);
	CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
}

void DeviceManager::Call(
	cl_kernel kernel,
	const vector<pair<const void*, size_t>> &arg_and_sizes,
	const cl_uint dim, const size_t *global_dim, const size_t *offset, const size_t *local_dim
) {
	cl_int result;

	auto name = GetKernelName(kernel);

	for (size_t i = 0; i < arg_and_sizes.size(); ++i) {
		auto &arg_and_size = arg_and_sizes[i];
        // LOG(INFO) << i << "\n";
		result = clSetKernelArg(kernel, i, arg_and_size.second, arg_and_size.first);
		CHECK_EQ(result, CL_SUCCESS) << name.data() << ' ' << clewErrorString(result);
	}
	// TODO: a OpenCL call is required here
    // cl_ulong size;
    // result = clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
	// CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
    // LOG(INFO) << size;
	result = clEnqueueNDRangeKernel( command_queue_, kernel, dim, nullptr, global_dim, local_dim, 0, nullptr, nullptr );
	CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
}

void DeviceManager::Finish()
{
	cl_int result;
	result = clFinish( command_queue_ );
	CHECK_EQ(result, CL_SUCCESS) << clewErrorString(result);
}
