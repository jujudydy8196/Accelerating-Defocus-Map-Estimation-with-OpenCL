#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

typedef unsigned char uchar;

// [2*r+1 2*r+1] box filter
void boxfilter(const float* I, float* Out, int width, int height, int r);

// A 3x3 Matrix
class Matrix3f
{
public:
	float a11;
	float a12;
	float a13;
	float a21;
	float a22;
	float a23;
	float a31;
	float a32;
	float a33;
};

Matrix3f invMat3(Matrix3f& A);

// Guided filter
class guided_filter
{
public:
	guided_filter(const float* I_guided, const int width, const int height, const int radius, const float eps);
	~guided_filter();
	void run(const float* input, float* output);
	void run_mask(float* input, float* output, uchar* mask);
// private:
	int _width;
	int _height;
	int numPixel;
	int r;
	// Image
	float* Ir;
	float* Ig;
	float* Ib;
	// Mean
	float* mean_Ir;
	float* mean_Ig;
	float* mean_Ib;
	// invSigma
	Matrix3f* invSigma;
	// Normalize term
	float* N;

};

// Full image range filter
class fullimage_rangefilter
{
public:
	fullimage_rangefilter(uchar* I_guided, const int width, const int height, const float sigma);
	~fullimage_rangefilter();
	void run(float* input, float* output);
	
private:
	int _width;
	int _height;
	float* Th;
	float* Tv;
	float* Al;
	float* Ar;
};

#endif