#ifndef VEC_H
#define VEC_H

#include <iostream>
#include "guidedfilter.h"

template <class T>
class Vec
{
public:
    Vec() : _data(NULL), _size(0) {}
    Vec( size_t size )
    {
        _size = size;
        _data = new T[_size];
    }
    Vec( const T* const data, size_t size )
    {
        _size = size;
        _data = new T[_size];
        copy( data );
    }
//    Vec( const Vec& v ): Vec( v._data, v._size ) {}
    Vec( const Vec& v ) {
		_size = v._size;
		_data = new T[_size];
		copy( v._data );
	}
    ~Vec() { reset(); }

    void reset()
    {
        delete [] _data;
        _data = NULL;
        _size = 0;
    }

    const Vec& operator=( const T* const data )
    {
        copy( data );
    }
    const Vec& operator=( const Vec& v )
    {
        if( _size != v._size ){
            delete [] _data;
            _size = v._size;
            _data = new T[_size];
        }
        copy( v._data );
    }

    T& operator[]( size_t i )
    {
        if( i >= _size ){
            std::cerr << "index exceed _size\n";
            // return T(0);
        }
        return _data[i];
    }
    const T& operator[]( size_t i ) const
    { 
        if( i >= _size ){
            std::cerr << "index exceed _size\n";
            // return T(0);
        }
        return _data[i];
    }

    static float dot( const Vec& v1, const Vec& v2 )
    {
        if( v1._size != v2._size ){
            std::cerr << "dot size mismatch.\n";
            return 0;
        }

        float sum = 0;
        for( size_t i = 0; i < v1._size; ++i ){
            sum += (float)v1[i] * (float)v2[i];
//			cout<<"v1 "<<v1[i]<<" "<<"v2 "<<v2[i]<<" sum "<<sum<<" "<<endl;
        }
        return sum;
    }
    // result = a1 * v1 + a2 * v2;
    static void add( Vec& result, const Vec& v1, const Vec& v2, float a1 = 1, float a2 = 1 )
    {
        for( size_t i = 0; i < result._size; ++i ){
            result[i] = a1 * v1[i] + a2 * v2[i];
        }
    }
    // result = v1 .* v2;
    static void multiply( Vec& result, const Vec& v1, const Vec& v2 )
    {
        for( size_t i = 0; i < result._size; ++i ){
            result[i] = v1[i] * v2[i];
        }
    }
    // result = v1 ./ v2;
    static void divide( Vec& result, const Vec& v1, const Vec& v2 )
    {
        for( size_t i = 0; i < result._size; ++i ){
            if( v2[i] < 0.01 && v2[i] > -0.01 ) result[i] = 0;
            else result[i] = v1[i] / v2[i];
        }
    }
    // result = a * v
    static void scalorMultiply( Vec& result, float a, const Vec& v )
    {
        for( size_t i = 0; i < result._size; ++i ){
            result[i] = a * v[i];
        }
    }

    size_t getSize() const
    {
        return _size;
    }
    const T* getPtr() const
    {
        return _data;
    }
    T* getPtr()
    {
        return _data;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec& v)
    {
        size_t size = v._size;
        os << "[ ";
        for( size_t i = 0; i < size; ++i ){
            os << v._data[i] << ' ';
        }
        os << ']';
        return os;
    }

private:
    void copy( const T* const data )
    {
        for( size_t i = 0; i < _size; ++i ){
            _data[i] = data[i];
        }
    }

    T*      _data;
    size_t  _size;
};

class LaplaMat
{
public:
    LaplaMat(const uchar* I_ori, const size_t width, const size_t height, const size_t r);
    void run(float* Lp, const float* p, const float lambda) const;
    ~LaplaMat();
private:
    guided_filter* _gf;
    float* _I_ori;
    size_t _r;
    size_t _width;
    size_t _height; 
};

LaplaMat::LaplaMat(const float* I_ori, const size_t width, const size_t height, const size_t r):_r(r), _width(width), _height(height) {
    _gf = new guided_filter(I_ori, width, height, r, 0.00001);
    int numPixel = _width * _height;
    _I_ori = new float[numPixel];
    for(int i = 0; i < numPixel; ++i) {
        _I_ori[i] = I_ori[i];
    }
}

void LaplaMat::run(float* Lp, const float* p, const float lambda) const {
    int numWinPixel, numPixel;
    numWinPixel = _r*_r;
    numPixel = _width * _height;
    float* tmpI = new float[numPixel];
    _gf->run(p, tmpI);
    
    for(int i = 0; i < numPixel; ++i) {
            //I-W*I
            //tmpI[i] = (float)_I_ori[i] - tmpI[i];
			// of1<<tmpI[i]<<" ";
			tmpI[i] = p[i] - tmpI[i];
			// of2<<tmpI[i]<<" ";
            //L = |w|*(I-W)     //L = lamda*L
            Lp[i] = (float)lambda * (float)numWinPixel * tmpI[i];
			//of2<<Lp[i]<<" ";
    }

    delete [] tmpI;
}

LaplaMat::~LaplaMat() {
    delete _gf;
}

#endif
