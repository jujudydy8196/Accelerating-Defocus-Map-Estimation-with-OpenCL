#include <iostream>
#include <string>
#include "fileIO.h"
#include "defocus.h"
#include "math.h"

// #define pi = 3.14;

using namespace std;

void g1x(float* g, int* x, int* y, float std, int w);
void filter(float* gim, float* g , uchar* I, int width, int height, int w);
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

	float gim[25];
	float g[9] = {0,1,2,3,4,5,6,7,8};
	uchar I[25] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
	filter( gim, g, I, 5, 5, 1 );
	for( size_t i = 0; i < 5; ++i ){
		for( size_t j = 0; j < 5; ++j ){
			cout << gim[i+j*5] << ' ';
		}
		cout << endl;
	}

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

void filter(float* gim, float* g , uchar* I, int width, int height, int w) {
	// cout << " w: " << w << endl;
	for(int i=0; i<height; i++) {
		for( int j=0; j<width; j++) {
			int count=0;
			float sum=0;
			for(int x=0; x<2*w+1; x++) {
				for (int y=0; y<2*w+1; y++) {
					if ((i-w+y)<0 || (i-w+y)>=height)
						continue;
					else if((j-w+x)<0 || (j-w+x)>=width)
						continue;
					else {
						sum +=( (float)I[(i-w+y)*width+(j-w+x)] * g[y*(2*w+1)+x]);
						// cout << "x: " << x << " y: " << y << " " << "i: " << i << " j: " << j ;
						// cout << " I: " <<(float)I[(i-w+y)*width+(j-w+x)]/255.0 << " g: " << g[y*(2*w+1)+x]<< " " << endl;
						// cout << " count: " << count << endl;
						count++;
					}
				}
			}
			gim[i*width+j] = sum;//count;
			// cout << "r: " << sum << " " << count << " " << sum/count <<" " << 
			// cout << gim[i*width+j] << " ";
		}
		// cout << endl;
	}
}