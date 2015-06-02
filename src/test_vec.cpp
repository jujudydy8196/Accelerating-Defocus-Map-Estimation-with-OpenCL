#include <iostream>
#include <vector>
#include "vec.h"
using namespace std;

void conjgrad( const vector< Vec<float> >&, const Vec<float>&, Vec<float>& );
void multiply( Vec<float>&, const vector< Vec<float> >&, const Vec<float>& );

int main()
{
    Vec<float> a(3), b(3), c(3), d(3);
    vector< Vec<float> > A( 3, Vec<float>(3) );
    a[0] = 1;
    a[1] = 3;
    a[2] = 2;

    // b[0] = 3;
    // b[1] = 5;
    // b[2] = 6;

    A[0][0] = 1;
    A[0][1] = 2;
    A[0][2] = 3;
    A[1][0] = 2;
    A[1][1] = 3;
    A[1][2] = 4;
    A[2][0] = 3;
    A[2][1] = 4;
    A[2][2] = 7;

    cout << "a: " << a << endl;
    cout << "b: " << b << endl;
    cout << "A:\n";
    for( size_t i = 0; i < 3; ++i ){
        cout << A[i] << endl;
    }

    multiply( b, A, a );
    cout << "A*a: " << b << endl;
    conjgrad( A, b, c );
    cout << "c: " << c << endl; 
    multiply( d, A, c );
    cout << "A*c: " << d << endl;
    /*cout << "b: " << b << endl;
    Vec<float> tmp(3);
    Vec<float>::add( a, a, b, 1, 1.5 );
    cout << "a=a+b: " << a << endl;
    Vec<float>::multiply( tmp, a, b );
    cout << "a*b: " << tmp << endl;
    cout << "a dot b: " << Vec<float>::dot( a, b ) << endl;*/
}

void conjgrad( const vector< Vec<float> >& A, const Vec<float>& estimate, Vec<float>& x )
{
    cout << "in conjgrad\n";
    size_t size = estimate.getSize();
    Vec<float> r( estimate ), p( estimate ), Ap( size );
    float rsold = Vec<float>::dot( r, r ), alpha = 0.0, rsnew = 0.0;

    for( size_t i = 0; i < 1000000; ++i ){
        // Ap = A * p
        cout << i << ' ' << rsold << endl;
        multiply( Ap, A, p );
        alpha = rsold / Vec<float>::dot( p, Ap );
        Vec<float>::add( x, x, p, 1, alpha );
        Vec<float>::add( r, r, Ap, 1, -alpha );
        rsnew = Vec<float>::dot( r, r );
        if( rsnew < 1e-20 ) break;
        Vec<float>::add( p, r, p, 1, rsnew/rsold );
        rsold = rsnew;
    }
}

void multiply( Vec<float>& result , const vector< Vec<float> >& A, const Vec<float>& v )
{
    size_t size = v.getSize();
    for( size_t i = 0; i < size; ++i ){
        result[i] = Vec<float>::dot( A[i], v );
    }
}

