#include <iostream>
#include "math.h"

#define pi = 3.14;

using namespace std;

void g1x(float* g, int* x, int* y, float std, int w);

int main(){


	int w=  (2*ceil(2* 0.5))+1;
	cout << "w: " << w << endl;
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

	float* g = new float[(int)pow(2*w+1,2)];
	g1x(g, x1, y1, 0.5, 3);
	for (int r=0; r<2*w+1; r++) {
		for (int c=0; c<2*w+1; c++) {
			cout << g[r*w+c] << " " ;
		}
		cout << endl;		
	}

	int* x2 = new int[(int)pow(2*w+1+2,2)];


	for (int i=0; i<2*w+1+2; i++) {
		for (int j=0; j<2*w+1+2; j++) {
			if (i==0 && j==0)
				x2[0] = x1[0];
			else if (i==0 && j==2*w+1+1)
				x2[j] = x1[j-2];
			else if (j==0 && i==2*w+1+1)
				x2[(2*w+1+2)*w] = x1[(2*w+1)*w];
			else if (j==2*w+1+1 && i==2*w+1+1)
				x2[(int)pow((2*w+1+2),2)-1] = x1[(int)pow(2*w+1,2)];
			else if (i==0)
				x2[j] = x1[j-1];
			else if (j==0)
				x2[i*(2*w+1+2)] = x1[i*(2*w+1)];
			else if (i==2*w+1+1)
				x2[(2*w+1+2)*i+j] = x1[(2*w+1)*i+j];
			else if (j==2*w+1+1)
				x2[(2*w+1+2)*i+j] = x1[(2*w+1)*i+j];
			else
				x2[(2*w+1+2)*i+j] = x1[(2*w+1+2)*(i-1)+(j-1)];
			cout << x2[(2*w+1+2)*i+j] << " " ;
		}
		cout << endl;
	}
	
}

void g1x(float* g, int* x, int* y, float std, int w) {
	float squareStd = pow(std,2);
	for (int i=0; i<pow(2*w+1,2); i++)
		g[i] = -(x[i]/(2*3.14*pow(squareStd,2)))* exp(-(pow(x[i],2)+pow(y[i],2))/(2*squareStd));
// g = -(x./(2*pi*s1sq.^2)) .* exp(-(x.^2 + y.^2)./(2*s1sq)); 
}