#include <iostream>
#include <string>
#include "fileIO.h"
#include "math.h"

#define pi = 3.14;

using namespace std;

void g1x(float* g, int* x, int* y, float std, int w);
void filter(float* gim, int* g , float* I, int width, int height, int w);
int main(){
	string name = "test_out.pgm";
	size_t wt = 20, ht = 50;
	uchar* testI = new uchar[wt*ht];

	for( size_t i = 0; i < wt; ++i ){
		for( size_t j = 0; j < ht; ++j ){
			testI[j*wt+i] = j*4;
		}
	}
	writePGM( testI, wt, ht, name );


	int w=  1;//(2*ceil(2* 0.5))+1;
	cout << "w: " << w << endl;
	int* x1 = new int[(int)pow(2*w+1,2)];
	int* y1 = new int[(int)pow(2*w+1,2)];
	cout << "x1: " << endl;
	for (int r=0; r<2*w+1; r++) {
		for (int c=0; c<2*w+1; c++) {
			x1[r*(2*w+1)+c] = -w+c;
			cout << x1[r*(2*w+1)+c] << " " ;
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
		cout << endl;		

	float* g = new float[(int)pow(2*w+1,2)];
	// g1x(g, x1, y1, 0.5, 3);
	for (int r=0; r<2*w+1; r++) {
		for (int c=0; c<2*w+1; c++) {
			g[r*(2*w+1)+c] = r*(2*w+1)+c;
			cout << g[r*(2*w+1)+c] << " " ;
		}
		cout << endl;		
	}
		cout << endl;		

	float* gim = new float[(int)pow(2*w+1,2)];

	filter(gim,x1,g,2*w+1,2*w+1,w);

	delete []x1;
	delete []y1;
	delete []gim;
	delete []g;

	// for (int i=0; i<pow(2*w+1,2); i++)
	// 	free(x1[i]);
	// free(x1);
	// for (int i=0; i<pow(2*w+1,2); i++)
	// 	free(x2[i]);
	// free(x2);
	// for (int i=0; i<pow(2*w+1,2); i++)
	// 	free(g[i]);
	// free(g);
	// for (int i=0; i<pow(2*w+1,2); i++)
	// 	free(gim[i]);
	// free(gim);
}

void g1x(float* g, int* x, int* y, float std, int w) {
	float squareStd = pow(std,2);
	for (int i=0; i<pow(2*w+1,2); i++)
		g[i] = -(x[i]/(2*3.14*pow(squareStd,2)))* exp(-(pow(x[i],2)+pow(y[i],2))/(2*squareStd));
// g = -(x./(2*pi*s1sq.^2)) .* exp(-(x.^2 + y.^2)./(2*s1sq)); 
}

void filter(float* gim, int* g , float* I, int width, int height, int w) {
	// cout << " w: " << w << endl;
	for(int i=0; i<height; i++) {
		for( int j=0; j<width; j++) {
			int count=0;
			float sum=0;
			for(int y=0; y<2*w+1; y++) {
				for (int x=0; x<2*w+1; x++) {
					if ((i-w+y)<0 || (i-w+y)>=height)
						continue;
					else if((j-w+x)<0 || (j-w+x)>=width)
						continue;
					else {
						sum +=( I[(i-w+y)*width+(j-w+x)] * g[y*(2*w+1)+x]);
						// cout << "x: " << x << " y: " << y << " " << "i: " << i << " j: " << j ;
						cout << " I: " << I[(i-w+y)*width+(j-w+x)] << " g: " << g[y*(2*w+1)+x]<< " " << endl;
						// cout << " count: " << count << endl;
						count++;
					}
				}
			}
			gim[i*width+j] = sum/count;
			cout << "r: "<< gim[i*width+j] << endl;
		}
		// cout << endl;
	}
}