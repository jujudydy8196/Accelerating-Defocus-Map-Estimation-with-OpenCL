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
            if( x < 0 || x >= width ) continue;
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
    __global const float *meanR,
    __global const float *meanG,
    __global const float *meanB,
    __global const float *varRR,
    __global const float *varRG,
    __global const float *varRB,
    __global const float *varGG,
    __global const float *varGB,
    __global const float *varBB,
    __global float *invSigma,
    const int size
)
{
    const float eps = 0.009;
    size_t id = get_global_id(0);

    if( id < size ){
        float a11 = varRR[id] - meanR[id] * meanR[id] + eps;
        float a12 = varRG[id] - meanR[id] * meanG[id];
        float a13 = varRB[id] - meanR[id] * meanB[id];
        float a21 = a12;
        float a22 = varGG[id] - meanG[id] * meanG[id] + eps;
        float a23 = varGB[id] - meanG[id] * meanB[id];
        float a31 = a13;
        float a32 = a23;
        float a33 = varBB[id] - meanB[id] * meanB[id] + eps;

        float a22a33_a23a32 = a22*a33 - a23*a32;
        float a13a32_a12a33 = a13*a32 - a12*a33;
        float a12a23_a13a22 = a12*a23 - a13*a22;
        float a23a31_a21a33 = a23*a31 - a21*a33;
        float a11a33_a13a31 = a11*a33 - a13*a31;
        float a13a21_a11a23 = a13*a21 - a11*a23;
        float a21a32_a22a31 = a21*a32 - a22*a31;
        float a12a31_a11a32 = a12*a31 - a11*a32;
        float a11a22_a12a21 = a11*a22 - a12*a21;
        float detA = a11*a22a33_a23a32 + a12*a23a31_a21a33 + a13*a21a32_a22a31;
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

__kernel void guidedFilterComputeAB(
    __global const float *meanR,
    __global const float *meanG,
    __global const float *meanB,
    __global const float *meanP,
    __global const float *varRP,
    __global const float *varGP,
    __global const float *varBP,
    __global float *a1,
    __global float *a2,
    __global float *a3,
    __global float *b,
    __global const float* invSigma,
    const int size
)
{
    size_t id = get_global_id(0);
    if( id < size ){
        float cov_Ir_p = varRP[id] - meanR[id] * meanP[id];
        float cov_Ig_p = varGP[id] - meanG[id] * meanP[id];
        float cov_Ib_p = varBP[id] - meanB[id] * meanP[id];

        a1[id] = cov_Ir_p*invSigma[9*id  ] + cov_Ig_p*invSigma[9*id+3] + cov_Ib_p*invSigma[9*id+6];
        a2[id] = cov_Ir_p*invSigma[9*id+1] + cov_Ig_p*invSigma[9*id+4] + cov_Ib_p*invSigma[9*id+7];
        a3[id] = cov_Ir_p*invSigma[9*id+2] + cov_Ig_p*invSigma[9*id+5] + cov_Ib_p*invSigma[9*id+8];

        b[id] = meanP[id] - a1[id]*meanR[id] - a2[id]*meanG[id] - a3[id]*meanB[id];
    }
}

__kernel void guidedFilterRunResult(
    __global float *q,
    __global const float *Ir,
    __global const float *Ig,
    __global const float *Ib,
    __global const float *meanA1,
    __global const float *meanA2,
    __global const float *meanA3,
    __global const float *meanB,
    const int size
)
{
    size_t id = get_global_id(0);

    if( id < size ){
        q[id] = Ir[id]*meanA1[id] + Ig[id]*meanA2[id] + Ib[id]*meanA3[id] + meanB[id];
    }
}