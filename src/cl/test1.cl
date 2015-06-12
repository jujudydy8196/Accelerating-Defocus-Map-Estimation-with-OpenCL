__kernel void test1(
    __global int *out
)
{
    int id = get_global_id(0);
    out[id] = id;
}
