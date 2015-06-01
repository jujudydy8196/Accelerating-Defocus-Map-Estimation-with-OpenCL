#include "defocus.h"

uchar* defocusEstimation(uchar* I, uchar* edge, float std, float lamda, float maxBlur) {
	// std :the standard devitation reblur gaussian1, typically std=[0.8:1]

	float std1= std;
	float std2= std1 * 1.5;
	int w=  (2*ceil(2* std1))+1;
	int* x1 = new int[(int)pow(2*w+1,2)];
	int* y1 = new int[(int)pow(2*w+1,2)];
	cout << "x1: " << endl;
	for (int r=0; r<2*w+1; r++) {
		for (int c=0; c<2*w+1; c++) {
			x1[r*w+c] = -w+c;
			cout << x1[r*w+c] << " " ;
		}
		cout << endl;		
	}
	cout << "y1: " << endl;
	for (int r=0; r<2*w+1; r++) {
		for (int c=0; c<2*w+1; c++) {
			y1[r*w+c] = -w+r;
			cout << y1[r*w+c] << " " ;
		}
		cout << endl;		
	}
	float* gx1 = new float[(int)pow(2*w+1,2)];
	g1x(gx1,x1,y1,std1,w);
	float* gy1 = new float[(int)pow(2*w+1,2)];
	g1y(gy1,x1,y1,std1,w);

}

void g1x(float* g, int* x, int* y, float std, int w) {
	float squareStd = pow(std,2);
	for (int i=0; i<pow(2*w+1,2); i++)
		g[i] = -(x[i]/(2*pi*pow(squareStd,2)))* exp(-(pow(x[i],2)+pow(y[i],2))/(2*squareStd));
// g = -(x./(2*pi*s1sq.^2)) .* exp(-(x.^2 + y.^2)./(2*s1sq)); 
}

void g1y(float* g, int* x, int* y, float std, int w) {
	float squareStd = pow(std,2);
	for (int i=0; i<pow(2*w+1,2); i++)
		g[i] = -(y[i]/(2*pi*pow(squareStd,2)))* exp(-(pow(x[i],2)+pow(y[i],2))/(2*squareStd));
// g = -(y./(2*pi*s1sq.^2)) .* exp(-(x.^2 + y.^2)./(2*s1sq)); 

}

void filter(float* gim, float*g , uchar* I) {

}