#include <cmath>
#include <algorithm>
#include <iostream>
#include "guidedfilter.h"
using namespace std;

void boxfilter(const float* I, float* Out, int width, int height, int r) {

	float* tmp = new float[width*height];
	
	// cumulative sum over Y axis
	for(int x = 0; x < width; ++x) {
		// y = 0
		tmp[x] = I[x];
		// y > 0
		for(int y = 1; y < height; ++y) {
			tmp[y*width+x] = tmp[(y-1)*width+x] + I[y*width+x];
		}
	}

	// difference over Y axis
	// imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
	for(int y = 0; y <= r; ++y) {
	for(int x = 0; x < width; ++x) {
		Out[y*width+x] = tmp[(y+r)*width+x];
	}
	}
	// imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);
	for(int y = r+1; y < height-r; ++y) {
	for(int x = 0; x < width; ++x) {
		Out[y*width+x] = tmp[(y+r)*width+x] - tmp[(y-r-1)*width+x];
	}
	}
	// imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);
	for(int y = height-r; y < height; ++y) {
	for(int x = 0; x < width; ++x) {
		Out[y*width+x] = tmp[(height-1)*width+x] - tmp[(y-r-1)*width+x];
	}
	}

	// cumulative sum over X axis
	for(int y = 0; y < height; ++y) {
		// x = 0
		tmp[y*width] = Out[y*width];
		// x > 0
		for(int x = 1; x < width; ++x) {
			tmp[y*width+x] = tmp[y*width+x-1] + Out[y*width+x];
		}
	}
	// difference over X axis
	// imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);
	for(int x = 0; x <= r; ++x) {
	for(int y = 0; y < height; ++y) {
		Out[y*width+x] = tmp[y*width+x+r];
	}
	}
	// imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);
	for(int x = r+1; x < width-r; ++x) {
	for(int y = 0; y < height; ++y) {
		Out[y*width+x] = tmp[y*width+x+r] - tmp[y*width+x-r-1];
	}
	}
	// imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);
	for(int x = width-r; x < width; ++x) {
	for(int y = 0; y < height; ++y) {
		Out[y*width+x] = tmp[y*width+width-1] - tmp[y*width+x-r-1];
	}
	}

	delete [] tmp;
}

void boxfilter(const uchar* I, float* Out, int width, int height, int r) {

	float* tmp = new float[width*height];
	
	// cumulative sum over Y axis
	for(int x = 0; x < width; ++x) {
		// y = 0
		tmp[x] = I[x];
		// y > 0
		for(int y = 1; y < height; ++y) {
			tmp[y*width+x] = tmp[(y-1)*width+x] + I[y*width+x];
		}
	}

	// difference over Y axis
	// imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
	for(int y = 0; y <= r; ++y) {
	for(int x = 0; x < width; ++x) {
		Out[y*width+x] = tmp[(y+r)*width+x];
	}
	}
	// imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);
	for(int y = r+1; y < height-r; ++y) {
	for(int x = 0; x < width; ++x) {
		Out[y*width+x] = tmp[(y+r)*width+x] - tmp[(y-r-1)*width+x];
	}
	}
	// imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);
	for(int y = height-r; y < height; ++y) {
	for(int x = 0; x < width; ++x) {
		Out[y*width+x] = tmp[(height-1)*width+x] - tmp[(y-r-1)*width+x];
	}
	}

	// cumulative sum over X axis
	for(int y = 0; y < height; ++y) {
		// x = 0
		tmp[y*width] = Out[y*width];
		// x > 0
		for(int x = 1; x < width; ++x) {
			tmp[y*width+x] = tmp[y*width+x-1] + Out[y*width+x];
		}
	}
	// difference over X axis
	// imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);
	for(int x = 0; x <= r; ++x) {
	for(int y = 0; y < height; ++y) {
		Out[y*width+x] = tmp[y*width+x+r];
	}
	}
	// imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);
	for(int x = r+1; x < width-r; ++x) {
	for(int y = 0; y < height; ++y) {
		Out[y*width+x] = tmp[y*width+x+r] - tmp[y*width+x-r-1];
	}
	}
	// imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);
	for(int x = width-r; x < width; ++x) {
	for(int y = 0; y < height; ++y) {
		Out[y*width+x] = tmp[y*width+width-1] - tmp[y*width+x-r-1];
	}
	}

	delete [] tmp;
}

// 3x3 matrix inverse
Matrix3f invMat3(Matrix3f& A) {
	
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
	
	Matrix3f invA;
	invA.a11 = a22a33_a23a32*detA;
	invA.a12 = a13a32_a12a33*detA;
	invA.a13 = a12a23_a13a22*detA;
	invA.a21 = a23a31_a21a33*detA;
	invA.a22 = a11a33_a13a31*detA;
	invA.a23 = a13a21_a11a23*detA;
	invA.a31 = a21a32_a22a31*detA;
	invA.a32 = a12a31_a11a32*detA;
	invA.a33 = a11a22_a12a21*detA;

	return invA;
}

gray_guided_filter::gray_guided_filter(const float* I_guided, const int width, const int height, const int radius, const float eps) {

	numPixel = width*height;
	_width = width;
	_height = height;
	r = radius;
	I = new float[numPixel];
	mean_I = new float[numPixel];
	N = new float[numPixel];
	invSigma = new float[numPixel];

	float* tmp = new float[numPixel];
	float* var_I = new float[numPixel];

	for(int i = 0; i < numPixel; ++i) {
		I[i] = (I_guided[i]);
		tmp[i] = 1;
	}

	boxfilter(tmp, N, width, height, r);
	boxfilter(I, mean_I, width, height, r);
	for(int i = 0; i < numPixel; ++i) {
		mean_I[i] /= N[i];
	}

	for(int i = 0; i < numPixel; ++i) {	tmp[i] = I[i]*I[i]; }	boxfilter(tmp, var_I, width, height, r);

	for(int i = 0; i < numPixel; ++i) {
		invSigma[i] = 1.0/float(var_I[i]/N[i] - mean_I[i]*mean_I[i]);
		// cout << invSigma[i] << " ";
 	}

	delete [] tmp;
	delete [] var_I;
}


gray_guided_filter::~gray_guided_filter() {
	delete [] I;
	delete [] mean_I;
	delete [] invSigma;
	delete [] N;
}

void gray_guided_filter::run(const float* p, float* q) {

	float* mean_p = new float[numPixel];
	float* buffer1 = new float[numPixel];
	float* tmp = new float[numPixel];
	float* a1 = new float[numPixel];
	float* b = new float[numPixel];

	float cov_I_p;

	boxfilter(p, mean_p, _width, _height, r);
	for(int i = 0; i < numPixel; ++i) {	mean_p[i] /= N[i]; }

	// mean_Ic_p, c = {r, g, b}
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = I[i]*p[i]; }	boxfilter(tmp, buffer1, _width, _height, r);
	// cov_Ic_p, c = {r, g, b}
	for(int i = 0; i < numPixel; ++i) {
		cov_I_p = buffer1[i]/N[i] - mean_I[i]*mean_p[i];
		// Eqn. (14) in the paper;
		a1[i] = cov_I_p*invSigma[i];
		// Eqn. (15) in the paper;
		b[i] = mean_p[i] - a1[i]*mean_I[i] ;		// b
	}

	boxfilter(a1, buffer1, _width, _height, r);
	boxfilter(b, tmp, _width, _height, r);

	// Eqn. (16) in the paper;
	for(int i = 0; i < numPixel; ++i) {
		q[i] = (buffer1[i]*I[i] + tmp[i])/N[i];
		cout << buffer1[i] << " * " << I[i] << " + " <<tmp[i] << " / " << N[i] << endl;
	}

	delete [] mean_p;
	delete [] buffer1;
	delete [] tmp;
	delete [] a1;
	delete [] b;
}
guided_filter::guided_filter(const float* I_guided, const int width, const int height, const int radius, const float eps) {
	
	numPixel = width*height;
	_width = width;
	_height = height;
	r = radius;
	Ir = new float[numPixel];
	Ig = new float[numPixel];
	Ib = new float[numPixel];
	mean_Ir = new float[numPixel];
	mean_Ig = new float[numPixel];
	mean_Ib = new float[numPixel];
	invSigma = new Matrix3f[numPixel];
	N = new float[numPixel];

	float* tmp = new float[numPixel];
	float* var_I_rr = new float[numPixel];
	float* var_I_rg = new float[numPixel];
	float* var_I_rb = new float[numPixel];
	float* var_I_gg = new float[numPixel];
	float* var_I_gb = new float[numPixel];
	float* var_I_bb = new float[numPixel];

	for(int i = 0; i < numPixel; ++i) {
		Ir[i] = (I_guided[3*i  ]);
		Ig[i] = (I_guided[3*i+1]);
		Ib[i] = (I_guided[3*i+2]);
		tmp[i] = 1;
	}

	boxfilter(tmp, N, width, height, r);
	boxfilter(Ir, mean_Ir, width, height, r);
	boxfilter(Ig, mean_Ig, width, height, r);
	boxfilter(Ib, mean_Ib, width, height, r);
	for(int i = 0; i < numPixel; ++i) {
		mean_Ir[i] /= N[i];
		mean_Ig[i] /= N[i];
		mean_Ib[i] /= N[i];
	}


	//           rr, rg, rb
	//   Sigma = rg, gg, gb
	//           rb, gb, bb
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ir[i]*Ir[i]; }	boxfilter(tmp, var_I_rr, width, height, r);
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ir[i]*Ig[i]; }	boxfilter(tmp, var_I_rg, width, height, r);
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ir[i]*Ib[i]; }	boxfilter(tmp, var_I_rb, width, height, r);
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ig[i]*Ig[i]; }	boxfilter(tmp, var_I_gg, width, height, r);
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ig[i]*Ib[i]; }	boxfilter(tmp, var_I_gb, width, height, r);
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ib[i]*Ib[i]; }	boxfilter(tmp, var_I_bb, width, height, r);

	// Compute invSigma
	for(int i = 0; i < numPixel; ++i) {
		Matrix3f Sigma;
		Sigma.a11 = var_I_rr[i]/N[i] - mean_Ir[i]*mean_Ir[i];
		Sigma.a12 = var_I_rg[i]/N[i] - mean_Ir[i]*mean_Ig[i];
		Sigma.a13 = var_I_rb[i]/N[i] - mean_Ir[i]*mean_Ib[i];
		Sigma.a22 = var_I_gg[i]/N[i] - mean_Ig[i]*mean_Ig[i];
		Sigma.a23 = var_I_gb[i]/N[i] - mean_Ig[i]*mean_Ib[i];
		Sigma.a33 = var_I_bb[i]/N[i] - mean_Ib[i]*mean_Ib[i];
		Sigma.a21 = Sigma.a12;
		Sigma.a31 = Sigma.a13;
		Sigma.a32 = Sigma.a23;
		Sigma.a11 += eps;
		Sigma.a22 += eps;
		Sigma.a33 += eps;
		invSigma[i] = invMat3(Sigma);
	}
	ofstream outfile1("invSigma.txt");
	for(int i = 0; i < numPixel; ++i) {
		outfile1<<invSigma[i].a11<<" "<<invSigma[i].a12<<" "<<invSigma[i].a13<<endl;
		outfile1<<invSigma[i].a21<<" "<<invSigma[i].a22<<" "<<invSigma[i].a23<<endl;
		outfile1<<invSigma[i].a31<<" "<<invSigma[i].a32<<" "<<invSigma[i].a33<<endl;
		outfile1<<endl;
	}
	outfile1.close();

	delete [] tmp;
	delete [] var_I_rr;
	delete [] var_I_rg;
	delete [] var_I_rb;
	delete [] var_I_gg;
	delete [] var_I_gb;
	delete [] var_I_bb;
}

guided_filter::~guided_filter() {
	delete [] Ir;
	delete [] Ig;
	delete [] Ib;
	delete [] mean_Ir;
	delete [] mean_Ig;
	delete [] mean_Ib;
	delete [] invSigma;
	delete [] N;
}

void guided_filter::run(const float* p, float* q) {

	float* mean_p = new float[numPixel];
	float* buffer1 = new float[numPixel];
	float* buffer2 = new float[numPixel];
	float* buffer3 = new float[numPixel];
	float* tmp = new float[numPixel];
	float* a1 = new float[numPixel];
	float* a2 = new float[numPixel];
	float* a3 = new float[numPixel];
	float* b = new float[numPixel];

	float cov_Ir_p, cov_Ig_p, cov_Ib_p;

	boxfilter(p, mean_p, _width, _height, r);
	for(int i = 0; i < numPixel; ++i) {	mean_p[i] /= N[i]; }

	// mean_Ic_p, c = {r, g, b}
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ir[i]*p[i]; }	boxfilter(tmp, buffer1, _width, _height, r);
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ig[i]*p[i]; }	boxfilter(tmp, buffer2, _width, _height, r);
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ib[i]*p[i]; }	boxfilter(tmp, buffer3, _width, _height, r);
	// cov_Ic_p, c = {r, g, b}
	for(int i = 0; i < numPixel; ++i) {
		cov_Ir_p = buffer1[i]/N[i] - mean_Ir[i]*mean_p[i];
		cov_Ig_p = buffer2[i]/N[i] - mean_Ig[i]*mean_p[i];
		cov_Ib_p = buffer3[i]/N[i] - mean_Ib[i]*mean_p[i];
		// Eqn. (14) in the paper;
		a1[i] = cov_Ir_p*invSigma[i].a11 + cov_Ig_p*invSigma[i].a21 + cov_Ib_p*invSigma[i].a31;
		a2[i] = cov_Ir_p*invSigma[i].a12 + cov_Ig_p*invSigma[i].a22 + cov_Ib_p*invSigma[i].a32;
		a3[i] = cov_Ir_p*invSigma[i].a13 + cov_Ig_p*invSigma[i].a23 + cov_Ib_p*invSigma[i].a33;
		// Eqn. (15) in the paper;
		b[i] = mean_p[i] - a1[i]*mean_Ir[i] - a2[i]*mean_Ig[i] - a3[i]*mean_Ib[i];		// b
	}

	boxfilter(a1, buffer1, _width, _height, r);
	boxfilter(a2, buffer2, _width, _height, r);
	boxfilter(a3, buffer3, _width, _height, r);
	boxfilter(b, tmp, _width, _height, r);

	// Eqn. (16) in the paper;
	for(int i = 0; i < numPixel; ++i) {
		q[i] = (buffer1[i]*Ir[i] + buffer2[i]*Ig[i] + buffer3[i]*Ib[i] + tmp[i])/N[i];
	}

	delete [] mean_p;
	delete [] buffer1;
	delete [] buffer2;
	delete [] buffer3;
	delete [] tmp;
	delete [] a1;
	delete [] a2;
	delete [] a3;
	delete [] b;
}
void guided_filter::run_mask(float* p, float* q, uchar* mask) {
	float* mean_p = new float[numPixel];
	float* buffer1 = new float[numPixel];
	float* buffer2 = new float[numPixel];
	float* buffer3 = new float[numPixel];
	float* tmp = new float[numPixel];
	float* a1 = new float[numPixel];
	float* a2 = new float[numPixel];
	float* a3 = new float[numPixel];
	float* b = new float[numPixel];

	float cov_Ir_p, cov_Ig_p, cov_Ib_p;

	boxfilter(p, mean_p, _width, _height, r);
	for(int i = 0; i < numPixel; ++i) {	mean_p[i] /= N[i]; }

	// mean_Ic_p, c = {r, g, b}
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ir[i]*p[i]; }	boxfilter(tmp, buffer1, _width, _height, r);
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ig[i]*p[i]; }	boxfilter(tmp, buffer2, _width, _height, r);
	for(int i = 0; i < numPixel; ++i) {	tmp[i] = Ib[i]*p[i]; }	boxfilter(tmp, buffer3, _width, _height, r);
	// cov_Ic_p, c = {r, g, b}
	for(int i = 0; i < numPixel; ++i) {
		cov_Ir_p = buffer1[i]/N[i] - mean_Ir[i]*mean_p[i];
		cov_Ig_p = buffer2[i]/N[i] - mean_Ig[i]*mean_p[i];
		cov_Ib_p = buffer3[i]/N[i] - mean_Ib[i]*mean_p[i];
		// Eqn. (14) in the paper;
		a1[i] = cov_Ir_p*invSigma[i].a11 + cov_Ig_p*invSigma[i].a21 + cov_Ib_p*invSigma[i].a31;
		a2[i] = cov_Ir_p*invSigma[i].a12 + cov_Ig_p*invSigma[i].a22 + cov_Ib_p*invSigma[i].a32;
		a3[i] = cov_Ir_p*invSigma[i].a13 + cov_Ig_p*invSigma[i].a23 + cov_Ib_p*invSigma[i].a33;
		// Eqn. (15) in the paper;
		b[i] = mean_p[i] - a1[i]*mean_Ir[i] - a2[i]*mean_Ig[i] - a3[i]*mean_Ib[i];		// b
	}

	boxfilter(a1, buffer1, _width, _height, r);
	boxfilter(a2, buffer2, _width, _height, r);
	boxfilter(a3, buffer3, _width, _height, r);
	boxfilter(b, tmp, _width, _height, r);

	// Eqn. (16) in the paper;
	for(int i = 0; i < numPixel; ++i) {
		if (mask[i])  q[i] = p[i];
		else q[i] = (buffer1[i]*Ir[i] + buffer2[i]*Ig[i] + buffer3[i]*Ib[i] + tmp[i])/N[i];
	}

	delete [] mean_p;
	delete [] buffer1;
	delete [] buffer2;
	delete [] buffer3;
	delete [] tmp;
	delete [] a1;
	delete [] a2;
	delete [] a3;
	delete [] b;
}
