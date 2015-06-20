// sum array for each block
__kernel void vecSum(
    __global float *result,
    __global const float *v,
    __local float *buffer,
    const int size
)
{
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t lSize = get_local_size(0);

    if( gid < size ){
        buffer[lid] = v[gid];
    }
    else{
        buffer[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for( int offset = get_local_size(0) / 2; offset > 0; offset >>= 1 ){
        if( lid < offset ) buffer[lid] += buffer[lid+offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if( lid == 0 ) result[get_group_id(0)] = buffer[0];
}

// v1 copy to v2
__kernel void vecCopy(
    __global const float *v1,
    __global float *v2,
    const int size
)
{
    size_t id = get_global_id(0);

    if( id < size ){
        v2[id] = v1[id];
    }
}

// v1[i] = c
__kernel void vecCopyConstant(
    __global float *v,
    const float c,
    const int size
)
{
    size_t id = get_global_id(0);

    if( id < size ){
        v[id] = c;
    }
}

// result = a1 * v1 + a2 * v2;
__kernel void vecScalarAdd(
    __global float *result,
    __global const float *v1,
    __global const float *v2,
    const float a1,
    const float a2,
    const int size
)
{
    size_t id = get_global_id(0);

    if( id < size ){
        result[id] = a1 * v1[id] + a2 * v2[id];
    }
}

// result = v1 + v2;
__kernel void vecAdd(
    __global float *result,
    __global const float *v1,
    __global const float *v2,
    const int size
)
{
    size_t id = get_global_id(0);

    if( id < size ){
        result[id] = v1[id] + v2[id];
    }
}

// result = v1 .* v2;
__kernel void vecMultiply(
    __global float *result,
    __global const float *v1,
    __global const float *v2,
    const int size
)
{
    size_t id = get_global_id(0);
    size_t gSize = get_global_size(0);

    if( id < size ){
        result[id] = v1[id] * v2[id];
    }
}

// result = v1 ./ v2;
__kernel void vecDivide(
    __global float *result,
    __global const float *v1,
    __global const float *v2,
    const int size
)
{
    size_t id = get_global_id(0);
    size_t gSize = get_global_size(0);

        if( id < size ){
        result[id] = v1[id] / v2[id];
    }
}

// result = a * v;
__kernel void vecScalarMultiply(
    __global float *result,
    __global const float *v,
    const float a,
    const int size
)
{
    size_t id = get_global_id(0);
    size_t gSize = get_global_size(0);

    if( id < size ){
        result[id] = v[id] * a;
    }
}

__kernel void constructH(
    __global float *H,
    __global const float *estimate,
    const int size
)
{
    size_t id = get_global_id(0);

    if( id < size ){
        if( estimate[id] != 0 ) H[id] = 1;
        else H[id] = 0;
    }
}

// size should < 1024
__kernel void computeAlpha(
    __global float *alpha,
    __global const float *v,
    __local float *buffer,
    __global const float *rsold,
    const int size
)
{
    size_t id = get_global_id(0);

    vecSum( alpha, v, buffer, size );
    barrier(CLK_LOCAL_MEM_FENCE);
    if( id == 0 ){
        alpha[0] = rsold[0] / alpha[0];
    }
}

__kernel void computeXR(
    __global float *x,
    __global const float *p,
    __global float *r,
    __global const float *Ap,
    __global float *alpha,
	__global float *check,
    const int size
)
{
    size_t id = get_global_id(0);
	if(id==0) check[0] = alpha[0];

    vecScalarAdd( x, x, p, 1, alpha[0], size );
    vecScalarAdd( r, r, Ap, 1, -alpha[0], size );
}

// size < 1024
__kernel void computeRs(
    __global float *rsold,
    __global float *rsRatio,
    __global const float *v,
    __local float *buffer,
    const int size
)
{
    size_t id = get_global_id(0);

    vecSum( rsRatio, v, buffer, size );
    barrier(CLK_LOCAL_MEM_FENCE);

    if( id == 0 ){
        float new = rsRatio[0];
        rsRatio[0] = new / rsold[0];
        rsold[0] = new;
    }
}

__kernel void computeP(
    __global float *p,
    __global const float *r,
    __global const float *rsRatio,
    const int size
)
{
    size_t id = get_global_id(0);

    vecScalarAdd( p, r, p, 1, rsRatio[0], size );

}

__kernel void vecTest(
    __global float *result,
    __global const float *v1,
    __global const float *v2,
    const int size
)
{
    size_t id = get_global_id(0);

    for( int i = 0 ; i < 10; ++i ){
        vecAdd( result, v1, v2, size );
    }
}