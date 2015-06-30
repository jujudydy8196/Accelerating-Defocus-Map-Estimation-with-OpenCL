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

__kernel void boxfilterCumulateY(
    __global float *result,
    __global const float *image,
    __global float *cumulate, // buffer
    const int width,
    const int height,
    const int size,
    const int r
)
{
    size_t id = get_global_id(0); // x

    if( id < width ){
        float sum = image[id];
        cumulate[id] = sum;

        for( int index = width + id; index < size; index += width ){
            sum += image[index];
            cumulate[index] = sum;
        }

        int delta = r*width;
        int index = id;
        for( int y = 0; y <= r; index += width, ++y ){
            result[index] = cumulate[index+delta];
        }
        for( int y = r + 1; y < height - r; index += width, ++y ){
            result[index] = cumulate[index+delta] - cumulate[index-delta-width];
        }
        float tmp = cumulate[(height-1)*width+id];
        for( int y = height - r; y < height; index += width, ++y ){
            result[index] = tmp - cumulate[index-delta-width];
        }
    }
}

__kernel void boxfilterCumulateX(
    __global float *result,
    __global const float *image,
    __global float *cumulate, // buffer
    const int width,
    const int height,
    const int size,
    const int r
)
{
    size_t id = get_global_id(0);

    if( id < height ){
        int beginX = id * width;
        float sum = image[beginX];
        cumulate[beginX] = sum;

        for( int index = beginX+1; index < beginX+width; ++index ){
            sum += image[index];
            cumulate[index] = sum;
        }

        for( int x = beginX; x <= beginX + r; ++x ){
            result[x] = cumulate[x+r];
        }
        for( int x = beginX+r+1; x < beginX + width - r; ++x ){
            result[x] = cumulate[x+r] - cumulate[x-r-1];
        }
        float tmp = cumulate[beginX+width-1];
        for( int x = beginX + width - r; x < beginX + width; ++x ){
            result[x] = tmp - cumulate[x-r-1];
        }
    }
}
