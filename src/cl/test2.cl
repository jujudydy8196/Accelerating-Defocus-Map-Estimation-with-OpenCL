__kernel void test2(
    __global int *out
)
{
    int id = get_global_id(0);
    out[id] = out[id] * out[id];
}
