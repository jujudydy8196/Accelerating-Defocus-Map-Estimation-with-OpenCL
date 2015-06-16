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
    size_t gSize = get_global_size(0);

    if( id < size ){
        v2[id] = v1[id];
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

__kernel void vecTest(
    __global float *result,
    const int size
)
{
    size_t id = get_global_id(0);
    size_t gSize = get_global_size(0);
    size_t tmp = get_group_id(0);

    if( id < size ){
        result[id] = tmp;
    }
}
