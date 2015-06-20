// a = v1 dot v2
__kernel void dot(
    __global float a,
    __global const float *v1,
    __global const float *v2,
    __global int size
)
{
    int id = get_global_id(0);

    if( id < size ){
        a += v1[id] * v2[id];
    }
}

// v1 copy to v2
__kernel void copy(
    __global const float *v1,
    __global float *v2,
    __global int size
)
{
    int id = get_global_id(0);

    if( id < size ){
        v2[id] = v1[id];
    }
}

// result = a1 * v1 + a2 * v2;
__kernel void add(
    __global float *result,
    __global const float *v1,
    __global const float *v2,
    __global const float a1,
    __global const float a2,
    __global int size
)
{
    int id = get_global_id(0);

    if( id < size ){
        result[id] = a1 * v1[id] + a2 * v2[id];
    }
}

// result = v1 .* v2;
__kernel void multiply(
    __global float *result,
    __global const float *v1,
    __global const float *v2,
    __global int size
)
{
    int id = get_global_id(0);

    if( id < size ){
        result[id] = v1[id] * v2[id];
    }
}

// result = v1 ./ v2;
__kernel void divide(
    __global float *result,
    __global const float *v1,
    __global const float *v2,
    __global int size
)
{
    int id = get_global_id(0);

    if( id < size ){
        result[id] = v1[id] / v2[id];
    }
}

// result = a * v;
__kernel void divide(
    __global float *result,
    __global const float *v,
    __global const float a,
    __global int size
)
{
    int id = get_global_id(0);

    if( id < size ){
        result[id] = v1[id] * a;
    }
}
