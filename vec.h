#ifndef VEC_H
#define VEC_H

#include <iostream>

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
    Vec( const Vec& v )
    {
        Vec( v._data, v._size );
    }
    ~Vec() { reset(); }

    void reset()
    {
        delete _data;
        _data = NULL;
        _length = 0;
    }

    const Vec& operator=( const T* const data )
    {
        copy( data );
    }
    const Vec& operator=( const Vec& v )
    {
        if( _size != v._size ){
            delete _data [];
            _size = v._size;
            _data = new T[_size];
        }
        copy( v._data );
    }

    T& operator[]( size_t i )
    {
        if( i > _size ){
            std::cerr << "index exceed _size\n";
            return T();
        }
        return _data[i];
    }
    const T& operator[]( size_t i ) const
    { 
        if( i > _size ){
            std::cerr << "index exceed _size\n";
            return T();
        }
        return _data[i];
    }

    static double dot( const Vec& v1, const Vec& v2 )
    {
        if( v1._size != v2._size ){
            std::cerr << "dot size mismatch.\n";
            return 0;
        }

        double sum = 0;
        for( size_t i = 0; i < v1._size; ++i ){
            sum += v1[i] * v2[i];
        }
        return sum;
    }
    // result = a1 * v1 + a2 * v2;
    static void add( Vec& result, const Vec& v1, const Vec& v2, double a1 = 1, double a2 = 1 )
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
    // result = a * v
    static void scalorMultiply( Vec& result, double a, const Vec& v )
    {
        for( size_t i = 0; i < result._size; ++i ){
            result[i] = a * v[i];
        }
    }

    size_t getSize() const
    {
        return _size;
    }

private:
    void copy( const T* const data )
    {
        for( size_t i = 0; i < _size; ++i ){
            _data[i] = data[i];
        }
    }

    T*      _data;
    size_t  _length;
};

#endif