#ifndef FILE_IO_H
#define FILE_IO_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
using namespace std;

typedef unsigned char uchar;

uchar checkPixelRange(float x);
void saveFloat(float* data, const int width, const int height, string filename);
void sizePGM(int& width, int& height, string filename);
void readPGM(uchar* I, string filename);
void readPPM(uchar* I, string filename);
// Save as portable gray map
template <class T>
void writePGM(T* I, const int width, const int height, string filename)
{
	char* buffer = new char[height*width];
	for(int i = 0; i < height*width; ++i) buffer[i] = (char)I[i];
	ofstream ofs(filename.c_str(), ios::binary);
	ofs << "P5" << endl
		<< width << " " << height << endl
		<< 255 << endl;
	ofs.write(buffer, height*width);
	ofs.close();
	delete [] buffer;
}

// Save as portable pixel map (color image)
template <class T>
void writePPM(T* I, const int width, const int height, string filename)
{
	char* buffer = new char[height*width*3];
	for(int i = 0; i < height*width*3; ++i) buffer[i] = (char)I[i];
	ofstream ofs(filename.c_str(), ios::binary);
	ofs << "P6" << endl
		<< width << " " << height << endl
		<< "255" << endl;
	ofs.write(buffer, height*width*3);
	ofs.close();
	delete [] buffer;
}

#endif