#include "propagatecl.h"
#include "cl_helper.h"
#include "global.h"
#include "vec.h"

void propagatecl( const float* image, const float* estimatedBlur, const size_t w, const size_t h, const float lambda, const size_t radius, Vec<float>& result )
{
    loadKernels();

    int size = w * h;
    int width = w, height = h, r = radius;

    // allocate gpu memory
    auto d_image = device_manager->AllocateMemory(CL_MEM_READ_ONLY, 3*size*sizeof(float));
    auto d_H = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_r = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_p = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_Hp = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_Lp = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_Ap = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    // guided filter
    auto d_gf_R = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_G = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_B = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_meanR = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_meanG = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_meanB = device_manager->AllocateMemory(CL_MEM_READ_WRITE, size*sizeof(float));
    auto d_gf_invSigma = device_manager->AllocateMemory(CL_MEM_READ_WRITE, 9*size*sizeof(float));
    
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
    vector<pair<const void*, size_t>> arg_and_sizes;
    if( size % local_size1[0] )
        global_size1[0] = local_size1[0] * ( size / local_size1[0] + 1 );
    else global_size1[0] = size;
    if( w % local_size2[0] )
        global_size2[0] = local_size2[0] * ( w / local_size2[0] + 1 );
    else global_size2[0] = w;
    if( h % local_size2[1] )
        global_size2[1] = local_size2[1] * ( h / local_size2[1] + 1 );
    else global_size2[1] = h;

    cout << size << ' ' << global_size1[0] << endl;
    cout << w << ' ' << global_size2[0] << endl;
    cout << h << ' ' << global_size2[1] << endl;
    
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



    // conjgrad
    for( size_t i = 0; i < 1000; ++i ){
        // HFilter( Hp, p.getPtr(), H, size);           // Hp = H .* p
        // LM->run(Lp, p.getPtr(), lambda);
        // getAp( Ap.getPtr(), Hp, Lp, size);           // Ap = Hp + Lp
        // alpha = rsold / Vec<float>::dot( p, Ap );    // dot
        // Vec<float>::add( x, x, p, 1, alpha );        // add, but alpha
        // Vec<float>::add( r, r, Ap, 1, -alpha );      // add
        // rsnew = Vec<float>::dot( r, r );             // dot
        // Vec<float>::add( p, r, p, 1, rsnew/rsold );  // add, but rsnew/rsold
        // rsold = rsnew;
    }
    // device_manager->ReadMemory(result.getPtr(), *d_H.get(), size*sizeof(float));
    
}

void loadKernels()
{
    device_manager->GetKernel("vec.cl", "vecSum");
    device_manager->GetKernel("vec.cl", "vecCopy");
    device_manager->GetKernel("vec.cl", "vecScalarAdd");
    device_manager->GetKernel("vec.cl", "vecMultiply");
    device_manager->GetKernel("vec.cl", "vecDivide");
    device_manager->GetKernel("vec.cl", "vecScalarMultiply");
    device_manager->GetKernel("vec.cl", "constructH");

    device_manager->GetKernel("guidedfilter.cl", "boxfilter");
    device_manager->GetKernel("guidedfilter.cl", "guidedFilterRGB");
    device_manager->GetKernel("guidedfilter.cl", "guidedFilterInvMat");
    device_manager->GetKernel("guidedfilter.cl", "guidedFilterComputeAB");
}