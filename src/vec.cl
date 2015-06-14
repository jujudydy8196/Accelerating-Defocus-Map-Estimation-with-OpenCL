// a = v1 dot v2
__kernel void vecDot(
    __global float *a,
    __global const float *v1,
    __global const float *v2,
    const int size
)
{
    int id = get_global_id(0);

    if( id < size ){
        if( id == 0 ) *a = v1[id] * v2[id];
        else *a += v1[id] * v2[id];
    }
}

// v1 copy to v2
__kernel void vecCopy(
    __global const float *v1,
    __global float *v2,
    const int size
)
{
    int id = get_global_id(0);

    if( id < size ){
        v2[id] = v1[id];
    }
}

// result = a1 * v1 + a2 * v2;
__kernel void vecAdd(
    __global float *result,
    __global const float *v1,
    __global const float *v2,
    const float a1,
    const float a2,
    const int size
)
{
    int id = get_global_id(0);

    if( id < size ){
        result[id] = a1 * v1[id] + a2 * v2[id];
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
    int id = get_global_id(0);

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
    int id = get_global_id(0);

    if( id < size ){
        result[id] = v1[id] / v2[id];
    }
}

// result = a * v;
__kernel void vecScalorMultiply(
    __global float *result,
    __global const float *v,
    const float a,
    const int size
)
{
    int id = get_global_id(0);

    if( id < size ){
        result[id] = v[id] * a;
    }
}