#include <iostream>
#include <windows.h>	// include for minisecond timer
#include "fileIO.h"
#include "guidedfilter.h"
using namespace std;


void HFilter(uchar* Hp, const uchar* p, const int height, const int width) {
	int idx = 0;
	for(int y = 0; y < height; ++y) {
		for(int x = 0; x < width; ++x) {
			idx = y*width + x;
			if((int)p[idx] != 0) { Hp[idx] = 1;}
			else Hp[idx] = 0;
		}
	}
}

void lambda_LFilter(float* Lp, const uchar* I_ori, const uchar* p, const int height, const int width, const float lambda, const int r) {
	int idx, numWinPixel;
	numWinPixel = r*r;
	float* tmpI = new float[height*width];
	//W*I
	guided_filter gf(I_ori, width, height, r, 0.00001);
	gf.run(p, tmpI);
	
	for(int y = 0; y < height; ++y) {
		for(int x = 0; x < width; ++x) {
			idx = y*width + x;
			//I-W*I
			tmpI[idx] = (float)I_ori[idx] - tmpI[idx];
			//L = |w|*(I-W)     //L = lamda*L
			Lp[idx] = (float)lambda * (float)numWinPixel * tmpI[idx];
		}
	}
}

//Ap = (H + lamda_L)p = Hp + lamda_L*p
void getAp(float* Ap, const uchar* Hp, const float* Lp, const int numPixel) {
	for(int i = 0; i < numPixel; ++i) Ap[i] = (float)Hp[i] + Lp[i];	
}


int main(int argc, char** argv) {
	
	if(argc != 6) {
		cout << "Usage: propagate_blur.exe <original image> <sparse depth image> <full depth image> <lambda> <radius>" << endl;
		return -1;
	}
	
	uchar* I_sparse;
	uchar* I_full;
	uchar* I_ori;
	uchar* Hp;
	float* Lp;
	float* Ap;
	int width, height, numPixel, r;
	float lambda;

	sizePGM(width, height, argv[1]);
	lambda = atof(argv[4]);
	r = atof(argv[5]);
	numPixel = width*height;
	I_sparse = new uchar[numPixel];
	//I_full = new uchar[numPixel];
	int n = numPixel * 3;
	I_ori = new uchar[n];
	Hp = new uchar[10];
	Lp = new float[10];
	Ap = new float[10];
	readPGM(I_sparse, argv[2]);
	readPPM(I_ori, argv[1]);

	HFilter(Hp, I_sparse, height, width);
	lambda_LFilter(Lp, I_ori, I_sparse, height, width, lambda, r);
	getAp(Ap, Hp, Lp, numPixel);


	
	//writePGM(Hp, width, height, "check_read.pgm");
	
	system("PAUSE");
	return 0;
}