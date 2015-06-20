#include "propagatecl.h"
#include "propagate.h"
#include "cl_helper.h"
#include "global.h"
#include "vec.h"
#include <cmath>
#include <ctime>

clock_t timeStart[10];
double  totalTime[10];
inline void startT( size_t id = 0 )
{
    device_manager->Finish();
    timeStart[id] = clock();
}
inline void endT( size_t id = 0 )
{
    device_manager->Finish();
    totalTime[id] += clock() - timeStart[id];
}
inline void printT( size_t id = 0 )
{
    cout << totalTime[id] / CLOCKS_PER_SEC << endl;
}
inline void resetT( size_t id = 0 )
{
    totalTime[id] = 0;
}

void propagatecl( const float* image, const float* estimatedBlur, const size_t w, const size_t h, const float lambda, const size_t r, Vec<float>& result )
{
    loadKernels();

    int size = w * h;
    int width = w, height = h, radius = r;
    clock_t start, stop;

    // allocate gpu memory
    auto d_image = device_manager->AllocateMemory(CL_MEM_READ_ONLY, 3*size*sizeof(float));
    auto d_H = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_r = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_p = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_x = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_Hp = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_Lp = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_Ap = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_alpha = device_manager->AllocateMemory(CL_MEM_READ_WRITE, sizeof(float));
    auto d_rsold = device_manager->AllocateMemory(CL_MEM_READ_WRITE, sizeof(float));
    auto d_rsRatio = device_manager->AllocateMemory(CL_MEM_READ_WRITE, sizeof(float));
    // guided filter
    auto d_gf_R = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_G = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_B = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_meanR = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_meanG = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_meanB = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_Irr = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_Irg = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_Irb = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_Igg = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_Igb = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_Ibb = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_varIrr = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_varIrg = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_varIrb = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_varIgg = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_varIgb = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_varIbb = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_invSigma = device_manager->AllocateMemory(CL_MEM_READ_WRITE, 9*size*sizeof(float));
    // buffer
    auto d_meanP = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_tmp = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_varRP = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_varGP = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_varBP = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_a1 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_a2 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_a3 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_b = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_dotBuffer = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    // reference
    auto &d_rr = d_Hp;

    // write to gpu memory
    start = clock();
    device_manager->WriteMemory( image, *d_image.get(), 3*size*sizeof(float));
    device_manager->WriteMemory( estimatedBlur, *d_r.get(), size*sizeof(float));
    device_manager->WriteMemory( estimatedBlur, *d_p.get(), size*sizeof(float));
    stop = clock();
    cout << "write memory time: " << double( stop - start ) / CLOCKS_PER_SEC << endl;

    // decalre some variable
    cl_kernel kernel = device_manager->GetKernel("vec.cl", "constructH");
    size_t local_size1[1] = {1024};
    size_t global_size1[1] = {};
    size_t local_size2[2] = {32,32};
    size_t global_size2[2] = {};
    vector<pair<const void*, size_t>> arg_and_sizes;
    global_size1[0] = getGlobalSize( size, local_size1[0] );
    global_size2[0] = getGlobalSize( w, local_size2[0] );
    global_size2[1] = getGlobalSize( h, local_size2[1] );

    cout << size << ' ' << global_size1[0] << endl;
    cout << w << ' ' << global_size2[0] << endl;
    cout << h << ' ' << global_size2[1] << endl;
    
    start = clock();
    // constructH
    arg_and_sizes.push_back( pair<const void*, size_t>( d_H.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_r.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

    // guided filter initialize
    // guidedFilterRGB
    kernel = device_manager->GetKernel("guidedfilter.cl", "guidedFilterRGB");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_R.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_G.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_B.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_image.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );
    
    // mean rgb by boxfilter
    kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilter");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_meanR.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_R.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_meanG.get(), sizeof(cl_mem) );
    arg_and_sizes[1] = pair<const void*, size_t>( d_gf_G.get(), sizeof(cl_mem) );
    device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_meanB.get(), sizeof(cl_mem) );
    arg_and_sizes[1] = pair<const void*, size_t>( d_gf_B.get(), sizeof(cl_mem) );
    device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

	//          rr, rg, rb 
	// sigma =  rg, gg, gb
	//          rb, gb, bb
	// 
    kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
	//rr
	arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_Irr.get(), sizeof(cl_mem) ) ); // rr
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_R.get(), sizeof(cl_mem) ) ); // [1] r
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_R.get(), sizeof(cl_mem) ) ); // [2] r
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );
	//rg
    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_Irg.get(), sizeof(cl_mem) );    // rg
    arg_and_sizes[2] = pair<const void*, size_t>( d_gf_G.get(), sizeof(cl_mem) );         // [2] g
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );
	//rb
    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_Irb.get(), sizeof(cl_mem) );   //  rb
    arg_and_sizes[2] = pair<const void*, size_t>( d_gf_B.get(), sizeof(cl_mem) );        //  [2] b
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );
	//gg
    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_Igg.get(), sizeof(cl_mem) );   //  gg
    arg_and_sizes[1] = pair<const void*, size_t>( d_gf_G.get(), sizeof(cl_mem) );        //  [1] g
    arg_and_sizes[2] = pair<const void*, size_t>( d_gf_G.get(), sizeof(cl_mem) );        //  [2] g
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );
	//gb
    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_Igb.get(), sizeof(cl_mem) );   //  gb
    arg_and_sizes[2] = pair<const void*, size_t>( d_gf_B.get(), sizeof(cl_mem) );        //  [2] b
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );
	//bb
    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_Ibb.get(), sizeof(cl_mem) );   //  bb
    arg_and_sizes[1] = pair<const void*, size_t>( d_gf_B.get(), sizeof(cl_mem) );        //  [1] b
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );
	
    kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilter");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_varIrr.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_Irr.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );
	
    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_varIrg.get(), sizeof(cl_mem) );
    arg_and_sizes[1] = pair<const void*, size_t>( d_gf_Irg.get(), sizeof(cl_mem) );
    device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_varIrb.get(), sizeof(cl_mem) );
    arg_and_sizes[1] = pair<const void*, size_t>( d_gf_Irb.get(), sizeof(cl_mem) );
    device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_varIgg.get(), sizeof(cl_mem) );
    arg_and_sizes[1] = pair<const void*, size_t>( d_gf_Igg.get(), sizeof(cl_mem) );
    device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_varIgb.get(), sizeof(cl_mem) );
    arg_and_sizes[1] = pair<const void*, size_t>( d_gf_Igb.get(), sizeof(cl_mem) );
    device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_varIbb.get(), sizeof(cl_mem) );
    arg_and_sizes[1] = pair<const void*, size_t>( d_gf_Ibb.get(), sizeof(cl_mem) );
    device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

	//I'm not sure if this is the right way to release memory from gpu....
	// clReleaseMemObject(d_gf_Irr);
	// clReleaseMemObject(d_gf_Irg);
	// clReleaseMemObject(d_gf_Irb);
	// clReleaseMemObject(d_gf_Igg);
	// clReleaseMemObject(d_gf_Igb);
	// clReleaseMemObject(d_gf_Ibb);

	// invertSigma
    kernel = device_manager->GetKernel("guidedfilter.cl", "guidedFilterInvMat");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_meanR.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_meanG.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_meanB.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_varIrr.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_varIrg.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_varIrb.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_varIgg.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_varIgb.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_varIbb.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_invSigma.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

	// clReleaseMemObject(d_gf_varIrr);
	// clReleaseMemObject(d_gf_varIrg);
	// clReleaseMemObject(d_gf_varIrb);
	// clReleaseMemObject(d_gf_varIgg);
	// clReleaseMemObject(d_gf_varIgb);
	// clReleaseMemObject(d_gf_varIbb);
    
    // initialize rsold
    kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_rr.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_r.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_r.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

    // initialize x
    float fzero = 0;
    kernel = device_manager->GetKernel("vec.cl", "vecCopyConstant");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_x.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &fzero, sizeof(float) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );    

    int tmpSize = size;
    size_t tmpGlobalSize[1] = { global_size1[0] };
    // recursive sum
    while( tmpSize > local_size1[0] ){
        kernel = device_manager->GetKernel("vec.cl", "vecSum");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_dotBuffer.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rr.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( NULL, local_size1[0]*sizeof(float) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );

        if( tmpSize % local_size1[0] ) tmpSize = tmpSize / local_size1[0] + 1;
        else tmpSize = tmpSize / local_size1[0];
        tmpGlobalSize[0] = getGlobalSize( tmpSize, local_size1[0] );

        kernel = device_manager->GetKernel("vec.cl", "vecCopy");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_dotBuffer.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rr.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );
    }

    kernel = device_manager->GetKernel("vec.cl", "vecSum");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_rsold.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_rr.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( NULL, local_size1[0]*sizeof(float) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );

    stop = clock();
    cout << "init time: " << double( stop - start ) / CLOCKS_PER_SEC << endl;

    start = clock();

    // conjgrad
    clock_t lmStart;
    double lmCount = 0;
    double dotTime = 0;
    double elseTime = 0;
    float a1 = 0, a2 = 0;
    startT();
    for( size_t i = 0; i < 1000; ++i ){
        // cout << i << "\n";
        // HFilter( Hp, p.getPtr(), H, size);           // Hp = H .* p
        // lmStart = clock();
        kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Hp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_H.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );
        // elseTime += double( clock() - lmStart );

        lmStart = clock();
        // LM->run(Lp, p.getPtr(), lambda);
        //    guided filter run
        //       boxfilter
        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilter");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_meanP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_tmp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_R.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilter");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varRP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_tmp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_tmp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_G.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilter");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varGP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_tmp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_tmp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_B.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilter");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varBP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_tmp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "guidedFilterComputeAB");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_meanR.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_meanG.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_meanB.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_meanP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varRP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varGP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varBP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_a1.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_a2.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_a3.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_b.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_invSigma.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilter");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varRP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_a1.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_varGP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_a2.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_varBP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_a3.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_tmp.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_b.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "guidedFilterRunResult");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Lp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_R.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_G.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_B.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varRP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varGP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varBP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_tmp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        a1 = lambda * radius * radius;
        a2 = -a1;
        kernel = device_manager->GetKernel("vec.cl", "vecScalarAdd");
        arg_and_sizes.resize(0);
		//(lamda*r*r)(I-L)p = a1*p - a2*Lp
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Lp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Lp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &a1, sizeof(float) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &a2, sizeof(float) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        lmCount += double( clock() - lmStart );
        lmStart = clock();
        // getAp( Ap.getPtr(), Hp, Lp, size);           // Ap = Hp + Lp
        kernel = device_manager->GetKernel("vec.cl", "vecAdd");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Ap.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Hp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Lp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );
        // elseTime += double( clock() - lmStart );

        // alpha = rsold / Vec<float>::dot( p, Ap );    // dot
        lmStart = clock();
        auto &d_ApP = d_Hp;
        auto &d_sumBuffer = d_Lp;
        kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_ApP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Ap.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );


        int tmpSize = size;
        size_t tmpGlobalSize[1] = { global_size1[0] };
        // recursive sum
        while( tmpSize > local_size1[0] ){
            kernel = device_manager->GetKernel("vec.cl", "vecSum");
            arg_and_sizes.resize(0);
            arg_and_sizes.push_back( pair<const void*, size_t>( d_dotBuffer.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( d_ApP.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( NULL, local_size1[0]*sizeof(float) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
            device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );

            if( tmpSize % local_size1[0] ) tmpSize = tmpSize / local_size1[0] + 1;
            else tmpSize = tmpSize / local_size1[0];
            tmpGlobalSize[0] = getGlobalSize( tmpSize, local_size1[0] );

            kernel = device_manager->GetKernel("vec.cl", "vecCopy");
            arg_and_sizes.resize(0);
            arg_and_sizes.push_back( pair<const void*, size_t>( d_dotBuffer.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( d_ApP.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
            device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );
        }

        kernel = device_manager->GetKernel("vec.cl", "computeAlpha");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_alpha.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_ApP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( NULL, local_size1[0]*sizeof(float) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rsold.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );

        dotTime += double( clock() - lmStart );
        lmStart = clock();

        // Vec<float>::add( x, x, p, 1, alpha );        // add, but alpha
        // Vec<float>::add( r, r, Ap, 1, -alpha );      // add
        kernel = device_manager->GetKernel("vec.cl", "computeXR");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_x.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_r.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Ap.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_alpha.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_tmp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );
        // elseTime += double( clock() - lmStart );

        // rsnew = Vec<float>::dot( r, r );             // dot
        // auto &d_sumBuffer = d_Lp;
        lmStart = clock();
        kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rr.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_r.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_r.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        tmpSize = size;
        tmpGlobalSize[0] = global_size1[0];
        // recursive sum
        while( tmpSize > local_size1[0] ){
            kernel = device_manager->GetKernel("vec.cl", "vecSum");
            arg_and_sizes.resize(0);
            arg_and_sizes.push_back( pair<const void*, size_t>( d_dotBuffer.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( d_rr.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( NULL, local_size1[0]*sizeof(float) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
            device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );

            if( tmpSize % local_size1[0] ) tmpSize = tmpSize / local_size1[0] + 1;
            else tmpSize = tmpSize / local_size1[0];
            tmpGlobalSize[0] = getGlobalSize( tmpSize, local_size1[0] );

            kernel = device_manager->GetKernel("vec.cl", "vecCopy");
            arg_and_sizes.resize(0);
            arg_and_sizes.push_back( pair<const void*, size_t>( d_dotBuffer.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( d_rr.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
            device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );
        }

        kernel = device_manager->GetKernel("vec.cl", "computeRs");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rsold.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rsRatio.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rr.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( NULL, local_size1[0]*sizeof(float) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );
        dotTime += double( clock() - lmStart );

        lmStart = clock();
        float rsold;
        device_manager->ReadMemory(&rsold, *d_rsold.get(), sizeof(float));
        elseTime += double( clock() - lmStart );
        if( rsold < 1e-10  ) break;

        // Vec<float>::add( p, r, p, 1, rsnew/rsold );  // add, but rsnew/rsold
        // rsold = rsnew;
        kernel = device_manager->GetKernel("vec.cl", "computeP");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_r.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rsRatio.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        //printClMemory( 1, *d_rsold.get() );
    }
    endT();
    printT();

    stop = clock();
    cout << "conjgrad time: " << double( stop - start ) / CLOCKS_PER_SEC << endl;
    cout << "lm time: " << lmCount / CLOCKS_PER_SEC << endl;
    cout << "dot time: " << dotTime / CLOCKS_PER_SEC << endl;
    cout << "else time: " << elseTime / CLOCKS_PER_SEC << endl;

    start = clock();
    device_manager->ReadMemory(result.getPtr(), *d_x.get(), size*sizeof(float));
    stop = clock();
    cout << "read time: " << double( stop - start ) / CLOCKS_PER_SEC << endl;
}

void loadKernels()
{
    device_manager->GetKernel("vec.cl", "vecSum");
    device_manager->GetKernel("vec.cl", "vecCopy");
    device_manager->GetKernel("vec.cl", "vecCopyConstant");
    device_manager->GetKernel("vec.cl", "vecAdd");
    device_manager->GetKernel("vec.cl", "vecScalarAdd");
    device_manager->GetKernel("vec.cl", "vecMultiply");
    device_manager->GetKernel("vec.cl", "vecDivide");
    device_manager->GetKernel("vec.cl", "vecScalarMultiply");
    device_manager->GetKernel("vec.cl", "constructH");
    device_manager->GetKernel("vec.cl", "computeAlpha");
    device_manager->GetKernel("vec.cl", "computeXR");
    device_manager->GetKernel("vec.cl", "computeRs");
    device_manager->GetKernel("vec.cl", "computeP");

    device_manager->GetKernel("guidedfilter.cl", "boxfilter");
    device_manager->GetKernel("guidedfilter.cl", "guidedFilterRGB");
    device_manager->GetKernel("guidedfilter.cl", "guidedFilterInvMat");
    device_manager->GetKernel("guidedfilter.cl", "guidedFilterComputeAB");
    device_manager->GetKernel("guidedfilter.cl", "guidedFilterRunResult");
}

size_t getGlobalSize( int size, size_t local_size )
{
    if( size % local_size )
        return local_size * ( size / local_size + 1 );
    else return size;
}

void printClMemory( int size, cl_mem d )
{
	float *out = new float[size];
    device_manager->ReadMemory(out, d, size*sizeof(float));
	// ofstream outfile("check_Ir_cl.txt");
    for(size_t i = 0; i < size; ++i){
        cout << out[i] << ' ';
		//outfile<<out[i]<<' ';
    }
    cout << endl;
	//outfile.close();

    delete [] out;
}

void compareMemory( int size, float* cpp, cl_mem d, float threshold )
{
    float *cl = new float[size];
    int errorCount = 0, warningCount = 0;
    device_manager->ReadMemory(cl, d, size*sizeof(float));
    for(size_t i = 0; i < size; ++i){
        if( cpp[i] == cl[i] ) continue;
        else if( fabs(cl[i] - cpp[i]) <= threshold ) ++warningCount;
        else{
            cout << i << " : " << cpp[i] << ' ' << cl[i] << endl;
            ++errorCount;
        }
    }
    cout << errorCount << " / " << warningCount << " / " << size << endl;

    delete [] cl;
}
