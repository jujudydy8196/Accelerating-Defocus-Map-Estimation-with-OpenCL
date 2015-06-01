#include <iostream>
#include "vec.h"
using namespace std;

int main()
{
    Vec<float> a(3), b(3);
    a[0] = 1;
    a[1] = 3;
    a[2] = 2;

    b[0] = 3;
    b[1] = 5;
    b[2] = 6;

    cout << "a: " << a << endl;
    cout << "b: " << b << endl;
    Vec<float> tmp(3);
    Vec<float>::add( tmp, a, b, 1, 1.5 );
    cout << "a+b: " << tmp << endl;
    Vec<float>::multiply( tmp, a, b );
    cout << "a*b: " << tmp << endl;
    cout << "a dot b: " << Vec<float>::dot( a, b ) << endl;
}