#include "vec.h"

void propagate( uchar*, uchar*, size_t, double );
void constructH( uchar*, vec&, vec& );
void conjgrad( const Vec&, const Vec&, double, const Vec&, Vec& );

void propagate( uchar* image, uchar* estimatedBlur, size_t size, double lambda )
{
    Vec<double> H( size ), L, estimate( size ), x( size );
    constructH( estimatedBlur, H, estimate );
    constructL(  );
    conjgrad( H, L, lambda, estimate, x );
}

void constructH( uchar* estimatedBlur, vec& H, vec& estimate )
{
    size_t size = H.getSize();
    for( size_t i = 0; i < size; ++i ){
        estimate[i] = double( estimatedBlur[i] );
        if( estimatedBlur[i] ) H[i] = 0;
        else H[i] = 1;
    }  
}

void conjgrad( const Vec& H, const Vec& L, double lambda, const Vec& estimate, Vec& x )
{
    size_t size = H.size();
    Vec<double> r( estimate ), p( estimate ), Ap( size );
    double rsold = Vec<double>::dot( r, r ), alpha = 0.0, rsnew = 0.0;

    for( size_t i = 0; i < 1000000; ++i ){
        // Ap = A * p
        alpha = rsold / Vec<double>::dot( p, Ap );
        Vec<double>::add( x, x, p, 1, alpha );
        Vec<double>::add( r, r, Ap, 1, -alpha );
        rsnew = Vec<double>::dot( r, r );
        if( rsnew < 1e-20 ) break;
        Vec<double>::add( p, r, p, 1, rsnew/rsold );
        rsold = rsnew;
    }
}