__kernel void boxfilter(
    __global float *result,
    __global const float *image,
    const int width,
    const int height,
    const int r
)
{
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);
    float sum = 0;
    size_t count = 0;

    if( gx >= width || gy >= height ) return;

    for( int dy = -r; dy <= r; ++dy ){
        int y = gy + dy;
        if( y < 0 || y >= height ) continue;
        for( int dx = -r; dx <= r; ++dx ){
            int x = gx + dx;
            if( x < o || x >= width ) continue;
            sum += image[y*width+x];
            ++count;
        }
    }
    result[gy*width+gx] = sum / count;
}

__kernel void guidedFilterRGB(
    __global float *Ir,
    __global float *Ig,
    __global float *Ib,
    __global const float *image,
    const int size
)
{
    size_t id = get_global_id(0);

    if( id < size ){
        Ir[id] = image[3*id];
        Ig[id] = image[3*id+1];
        Ib[id] = image[3*id+2];
    }
}

__kernel void guidedFilterInvMat(
    __global float *meanR,
    __global float *meanG,
    __global float *meanB,
    __global float *varRR,
    __global float *varRG,
    __global float *varRB,
    __global float *varGG,
    __global float *varGB,
    __global float *varBB,
    __global float *invSigma,
    const int size
)
{
    const float eps = 0.009;
    size_t id = get_global_id(0);

    if( id < size ){
        float a11 = varRR - meanR * meanR + eps;
        float a12 = varRG - meanR * meanG;
        float a13 = varRB - meanR * meanB;
        float a21 = a12;
        float a22 = varGG - meanG * meanG + eps;
        float a23 = varGB - meanG * meanB;
        float a31 = a13;
        float a32 = a23;
        float a33 = varBB - meanB * meanB + eps;

        float a22a33_a23a32 = A.a22*A.a33 - A.a23*A.a32;
        float a13a32_a12a33 = A.a13*A.a32 - A.a12*A.a33;
        float a12a23_a13a22 = A.a12*A.a23 - A.a13*A.a22;
        float a23a31_a21a33 = A.a23*A.a31 - A.a21*A.a33;
        float a11a33_a13a31 = A.a11*A.a33 - A.a13*A.a31;
        float a13a21_a11a23 = A.a13*A.a21 - A.a11*A.a23;
        float a21a32_a22a31 = A.a21*A.a32 - A.a22*A.a31;
        float a12a31_a11a32 = A.a12*A.a31 - A.a11*A.a32;
        float a11a22_a12a21 = A.a11*A.a22 - A.a12*A.a21;
        float detA = A.a11*a22a33_a23a32 + A.a12*a23a31_a21a33 + A.a13*a21a32_a22a31;
        detA = 1.f/detA;
    
        invSigma[9*id  ] = a22a33_a23a32*detA;
        invSigma[9*id+1] = a13a32_a12a33*detA;
        invSigma[9*id+2] = a12a23_a13a22*detA;
        invSigma[9*id+3] = a23a31_a21a33*detA;
        invSigma[9*id+4] = a11a33_a13a31*detA;
        invSigma[9*id+5] = a13a21_a11a23*detA;
        invSigma[9*id+6] = a21a32_a22a31*detA;
        invSigma[9*id+7] = a12a31_a11a32*detA;
        invSigma[9*id+8] = a11a22_a12a21*detA;
    }
}