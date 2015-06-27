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

void propagatecl( const float* image, const float* estimatedBlur, const size_t w, const size_t h, const float lambda, const size_t ra, Vec<float>& result )
{
    loadKernels();

    int size = w * h;
    int width = w, height = h, radius = ra;

    // allocate gpu memory
    // buffer
    auto d_buffer0 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer1 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer2 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer3 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer4 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer5 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer6 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer7 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer8 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer9 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer10 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer11 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    // mem
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
    auto &d_gf_Irr = d_buffer0;
    auto &d_gf_Irg = d_buffer1;
    auto &d_gf_Irb = d_buffer2;
    auto &d_gf_Igg = d_buffer3;
    auto &d_gf_Igb = d_buffer4;
    auto &d_gf_Ibb = d_buffer5;
    auto &d_gf_varIrr = d_buffer6;
    auto &d_gf_varIrg = d_buffer7;
    auto &d_gf_varIrb = d_buffer8;
    auto &d_gf_varIgg = d_buffer9;
    auto &d_gf_varIgb = d_buffer10;
    auto &d_gf_varIbb = d_buffer11;
    // auto d_gf_N = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_invSigma = device_manager->AllocateMemory(CL_MEM_READ_WRITE, 9*size*sizeof(float));
    // gf run
    auto &d_meanP = d_buffer0;
    auto &d_rp = d_buffer1;
    auto &d_gp = d_buffer2;
    auto &d_bp = d_buffer3;
    auto &d_varRP = d_buffer4;
    auto &d_varGP = d_buffer5;
    auto &d_varBP = d_buffer6;
    auto &d_a1 = d_buffer7;
    auto &d_a2 = d_buffer8;
    auto &d_a3 = d_buffer9;
    auto &d_b = d_buffer10;
    auto &d_meanB = d_buffer11;
    // reference
    auto &d_dotBuffer = d_buffer0;
    auto &d_rr = d_buffer1;
    auto &d_tmp = d_buffer2;
    auto &d_ApP = d_Hp;    
    

    // write to gpu memory
    device_manager->WriteMemory( image, *d_image.get(), 3*size*sizeof(float));
    device_manager->WriteMemory( estimatedBlur, *d_r.get(), size*sizeof(float));
    device_manager->WriteMemory( estimatedBlur, *d_p.get(), size*sizeof(float));

    // decalre some variable
    cl_kernel kernel = device_manager->GetKernel("vec.cl", "constructH");
    size_t local_size1[1] = {1024};
    size_t global_size1[1] = {};
    size_t local_size2[2] = {32,32};
    size_t global_size2[2] = {};
    size_t local16[] = {16};
    vector<pair<const void*, size_t>> arg_and_sizes;
    global_size1[0] = getGlobalSize( size, 1024 );
    global_size2[0] = getGlobalSize( w, 32 );
    global_size2[1] = getGlobalSize( h, 32 );
    
    // constructH
    arg_and_sizes.push_back( pair<const void*, size_t>( d_H.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_r.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

    // guided filter initialize
    /*
    auto d_ones = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_box_tmp = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_box_buffer = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    float one = 1;
    size_t lll[1] = {16};

    // gf_N
    kernel = device_manager->GetKernel("vec.cl", "vecCopyConstant");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_ones.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &one, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

    kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_image.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_box_buffer.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local_size1 );

    kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
    arg_and_sizes[0] = pair<const void*, size_t>( d_ones.get(), sizeof(cl_mem) );
    arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
    device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, lll );
    */

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
    while( tmpSize > /*local_size1[0]*/ 1024 ){
        kernel = device_manager->GetKernel("vec.cl", "vecSum");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_dotBuffer.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rr.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( NULL,/* local_size1[0]*/1024*sizeof(float) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );

        if( tmpSize % 1024 /*local_size1[0]*/ ) tmpSize = tmpSize / 1024/*local_size1[0]*/ + 1;
        else tmpSize = tmpSize / 1024/*local_size1[0]*/;
        tmpGlobalSize[0] = getGlobalSize( tmpSize, 1024/*local_size1[0]*/ );

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
    arg_and_sizes.push_back( pair<const void*, size_t>( NULL, 1024/*local_size1[0]*/*sizeof(float) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );

    float a1 = 0, a2 = 0;
    int winNum = (2*radius+1)*(2*radius+1);
    // startT();
    for( size_t i = 0; i < 1000; ++i ){
        // cout << i << "\n";
        // HFilter( Hp, p.getPtr(), H, size);           // Hp = H .* p
        kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Hp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_H.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        // LM->run(Lp, p.getPtr(), lambda);
        //    guided filter run
        //       boxfilter
        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilter3");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_meanP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
        // arg_and_sizes.push_back( pair<const void*, size_t>( &winNum, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( NULL, (local_size2[0]+2*radius)*(local_size2[1]+2*radius)*sizeof(float) ) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_R.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_gp.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_gf_G.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_bp.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_gf_B.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilter3");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varRP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
        // arg_and_sizes.push_back( pair<const void*, size_t>( &winNum, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( NULL, (local_size2[0]+2*radius)*(local_size2[1]+2*radius)*sizeof(float) ) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_varGP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_gp.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_varBP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_bp.get(), sizeof(cl_mem) );
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

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilter3");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varRP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_a1.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
        // arg_and_sizes.push_back( pair<const void*, size_t>( &winNum, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( NULL, (local_size2[0]+2*radius)*(local_size2[1]+2*radius)*sizeof(float) ) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_varGP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_a2.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_varBP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_a3.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 2, global_size2, NULL, local_size2 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_meanB.get(), sizeof(cl_mem) );
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
        arg_and_sizes.push_back( pair<const void*, size_t>( d_meanB.get(), sizeof(cl_mem) ) );
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

        // getAp( Ap.getPtr(), Hp, Lp, size);           // Ap = Hp + Lp
        kernel = device_manager->GetKernel("vec.cl", "vecAdd");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Ap.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Hp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Lp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        // alpha = rsold / Vec<float>::dot( p, Ap );    // dot
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
        while( tmpSize > 1024/*local_size1[0]*/ ){
            kernel = device_manager->GetKernel("vec.cl", "vecSum");
            arg_and_sizes.resize(0);
            arg_and_sizes.push_back( pair<const void*, size_t>( d_dotBuffer.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( d_ApP.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( NULL, 1024/*local_size1[0]*/*sizeof(float) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
            device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );

            if( tmpSize % 1024/*local_size1[0]*/ ) tmpSize = tmpSize / 1024/*local_size1[0]*/ + 1;
            else tmpSize = tmpSize / 1024/*local_size1[0]*/;
            tmpGlobalSize[0] = getGlobalSize( tmpSize, 1024/*local_size1[0]*/ );

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
        arg_and_sizes.push_back( pair<const void*, size_t>( NULL, 1024/*local_size1[0]*/*sizeof(float) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rsold.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );


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

        // rsnew = Vec<float>::dot( r, r );             // dot
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
        while( tmpSize > 1024/*local_size1[0]*/ ){
            kernel = device_manager->GetKernel("vec.cl", "vecSum");
            arg_and_sizes.resize(0);
            arg_and_sizes.push_back( pair<const void*, size_t>( d_dotBuffer.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( d_rr.get(), sizeof(cl_mem) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( NULL, 1024/*local_size1[0]*/*sizeof(float) ) );
            arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
            device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );

            if( tmpSize % /*local_size1[0]*/1024 ) tmpSize = tmpSize / 1024/*local_size1[0]*/ + 1;
            else tmpSize = tmpSize / 1024/*local_size1[0]*/;
            tmpGlobalSize[0] = getGlobalSize( tmpSize, 1024/*local_size1[0]*/ );

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
        arg_and_sizes.push_back( pair<const void*, size_t>( NULL, 1024/*local_size1[0]*/*sizeof(float) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &tmpSize, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, tmpGlobalSize, NULL, local_size1 );

        float rsold;
        device_manager->ReadMemory(&rsold, *d_rsold.get(), sizeof(float));
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

    }
    device_manager->ReadMemory(result.getPtr(), *d_x.get(), size*sizeof(float));
}

void propagatecl2( const float* image, const float* estimatedBlur, const size_t w, const size_t h, const float lambda, const size_t ra, Vec<float>& result )
{
    loadKernels();

    int size = w * h;
    int width = w, height = h, radius = ra;

    // allocate gpu memory
    // buffer
    auto d_buffer0 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer1 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer2 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer3 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer4 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer5 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer6 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer7 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer8 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer9 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer10 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_buffer11 = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    // mem
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
    auto &d_gf_Irr = d_buffer0;
    auto &d_gf_Irg = d_buffer1;
    auto &d_gf_Irb = d_buffer2;
    auto &d_gf_Igg = d_buffer3;
    auto &d_gf_Igb = d_buffer4;
    auto &d_gf_Ibb = d_buffer5;
    auto &d_gf_varIrr = d_buffer6;
    auto &d_gf_varIrg = d_buffer7;
    auto &d_gf_varIrb = d_buffer8;
    auto &d_gf_varIgg = d_buffer9;
    auto &d_gf_varIgb = d_buffer10;
    auto &d_gf_varIbb = d_buffer11;
    auto d_gf_N = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_invSigma = device_manager->AllocateMemory(CL_MEM_READ_WRITE, 9*size*sizeof(float));
    // gf run
    auto &d_meanP = d_buffer0;
    auto &d_rp = d_buffer1;
    auto &d_gp = d_buffer2;
    auto &d_bp = d_buffer3;
    auto &d_varRP = d_buffer4;
    auto &d_varGP = d_buffer5;
    auto &d_varBP = d_buffer6;
    auto &d_a1 = d_buffer7;
    auto &d_a2 = d_buffer8;
    auto &d_a3 = d_buffer9;
    auto &d_b = d_buffer10;
    auto &d_meanB = d_buffer11;
    // boxfilter
    auto &d_ones = d_buffer0;
    auto d_box_tmp = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_box_buffer = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    // reference
    auto &d_dotBuffer = d_buffer0;
    auto &d_rr = d_buffer1;
    auto &d_tmp = d_buffer2;
    auto &d_ApP = d_Hp;    
    
    // write to gpu memory
    device_manager->WriteMemory( image, *d_image.get(), 3*size*sizeof(float));
    device_manager->WriteMemory( estimatedBlur, *d_r.get(), size*sizeof(float));
    device_manager->WriteMemory( estimatedBlur, *d_p.get(), size*sizeof(float));

    // decalre some variable
    cl_kernel kernel = device_manager->GetKernel("vec.cl", "constructH");
    size_t local_size1[1] = {1024};
    size_t global_size1[1] = {};
    size_t local_size2[2] = {32,32};
    size_t global_size2[2] = {};
    size_t local16[] = {16};
    vector<pair<const void*, size_t>> arg_and_sizes;
    global_size1[0] = getGlobalSize( size, local_size1[0] );
    global_size2[0] = getGlobalSize( w, local_size2[0] );
    global_size2[1] = getGlobalSize( h, local_size2[1] );
    
    // constructH
    arg_and_sizes.push_back( pair<const void*, size_t>( d_H.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_r.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

    // guided filter initialize
    // gf_N
    float one = 1;
    kernel = device_manager->GetKernel("vec.cl", "vecCopyConstant");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_ones.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &one, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

    kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_ones.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_box_buffer.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local_size1 );

    kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_N.get(), sizeof(cl_mem) );
    arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
    device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local16 );

    kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
    arg_and_sizes.resize(0);
    arg_and_sizes.push_back( pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_ones.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( d_box_buffer.get(), sizeof(cl_mem) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
    arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
    device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local_size1 );

    kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
    arg_and_sizes[0] = pair<const void*, size_t>( d_gf_N.get(), sizeof(cl_mem) );
    arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
    device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local16 );

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

    float a1 = 0, a2 = 0;
    int winNum = (2*radius+1)*(2*radius+1);
    // startT();
    for( size_t i = 0; i < 1000; ++i ){
        // cout << i << "\n";
        // HFilter( Hp, p.getPtr(), H, size);           // Hp = H .* p
        kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Hp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_H.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        // LM->run(Lp, p.getPtr(), lambda);
        //    guided filter run
        //       boxfilter
        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_box_buffer.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
        arg_and_sizes[0] = pair<const void*, size_t>( d_meanP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local16 );

        kernel = device_manager->GetKernel("vec.cl", "vecDivide");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_meanP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_meanP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_N.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("vec.cl", "vecMultiply");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_R.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_gp.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_gf_G.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        arg_and_sizes[0] = pair<const void*, size_t>( d_bp.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_gf_B.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_box_buffer.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
        arg_and_sizes[0] = pair<const void*, size_t>( d_varRP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local16 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
        arg_and_sizes[0] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_gp.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
        arg_and_sizes[0] = pair<const void*, size_t>( d_varGP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local16 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
        arg_and_sizes[0] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_bp.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
        arg_and_sizes[0] = pair<const void*, size_t>( d_varBP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local16 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "guidedFilterComputeAB2");
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
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_N.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_a1.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_box_buffer.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &width, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &height, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &radius, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
        arg_and_sizes[0] = pair<const void*, size_t>( d_varRP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local16 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
        arg_and_sizes[0] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_a2.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
        arg_and_sizes[0] = pair<const void*, size_t>( d_varGP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local16 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
        arg_and_sizes[0] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_a3.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
        arg_and_sizes[0] = pair<const void*, size_t>( d_varBP.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local16 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
        arg_and_sizes[0] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_b.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
        arg_and_sizes[0] = pair<const void*, size_t>( d_meanB.get(), sizeof(cl_mem) );
        arg_and_sizes[1] = pair<const void*, size_t>( d_box_tmp.get(), sizeof(cl_mem) );
        device_manager->Call( kernel, arg_and_sizes, 1, local_size1, NULL, local16 );

        kernel = device_manager->GetKernel("guidedfilter.cl", "guidedFilterRunResult");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Lp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_R.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_G.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_B.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varRP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varGP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_varBP.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_meanB.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        kernel = device_manager->GetKernel("vec.cl", "vecDivide");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Lp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Lp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_gf_N.get(), sizeof(cl_mem) ) );
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

        // getAp( Ap.getPtr(), Hp, Lp, size);           // Ap = Hp + Lp
        kernel = device_manager->GetKernel("vec.cl", "vecAdd");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Ap.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Hp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_Lp.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

        // alpha = rsold / Vec<float>::dot( p, Ap );    // dot
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

        // rsnew = Vec<float>::dot( r, r );             // dot
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

        float gpu_rsold;
        device_manager->ReadMemory(&gpu_rsold, *d_rsold.get(), sizeof(float));
        if( gpu_rsold < 1e-10  ) break;

        // Vec<float>::add( p, r, p, 1, rsnew/rsold );  // add, but rsnew/rsold
        // rsold = rsnew;
        kernel = device_manager->GetKernel("vec.cl", "computeP");
        arg_and_sizes.resize(0);
        arg_and_sizes.push_back( pair<const void*, size_t>( d_p.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_r.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( d_rsRatio.get(), sizeof(cl_mem) ) );
        arg_and_sizes.push_back( pair<const void*, size_t>( &size, sizeof(int) ) );
        device_manager->Call( kernel, arg_and_sizes, 1, global_size1, NULL, local_size1 );

    }
    device_manager->ReadMemory(result.getPtr(), *d_x.get(), size*sizeof(float));
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
    device_manager->GetKernel("guidedfilter.cl", "boxfilter2");
    device_manager->GetKernel("guidedfilter.cl", "boxfilter3");
    device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateX");
    device_manager->GetKernel("guidedfilter.cl", "boxfilterCumulateY");
    device_manager->GetKernel("guidedfilter.cl", "guidedFilterRGB");
    device_manager->GetKernel("guidedfilter.cl", "guidedFilterInvMat");
    device_manager->GetKernel("guidedfilter.cl", "guidedFilterComputeAB");
    device_manager->GetKernel("guidedfilter.cl", "guidedFilterComputeAB2");
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

void compareMemory( int size, cl_mem d1, cl_mem d2, float threshold )
{
    float *cl1 = new float[size];
    float *cl2 = new float[size];
    int errorCount = 0, warningCount = 0;
    device_manager->ReadMemory(cl1, d1, size*sizeof(float));
    device_manager->ReadMemory(cl2, d2, size*sizeof(float));
    for(size_t i = 0; i < size; ++i){
        if( cl1[i] == cl2[i] ) continue;
        else if( fabs(cl1[i] - cl2[i]) <= threshold ) ++warningCount;
        else{
            cout << i << " : " << cl1[i] << ' ' << cl2[i] << endl;
            ++errorCount;
        }
    }
    cout << errorCount << " / " << warningCount << " / " << size << endl;

    delete [] cl1;
    delete [] cl2;
}
