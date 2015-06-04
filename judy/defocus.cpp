#include "defocus.h"
#include "fileIO.h"

uchar* defocusEstimation(uchar* I, uchar* edge, float std, float lamda, float maxBlur, int width, int height) {
	// std :the standard devitation reblur gaussian1, typically std=[0.8:1]

	float std1= std;
	float std2= std1 * 1.5;
	int w=  (2*ceil(2* std1))+1;
	int* x1 = new int[(int)pow(2*w+1,2)];
	int* y1 = new int[(int)pow(2*w+1,2)];
	// cout << "x1: " << endl;
	for (int r=0; r<2*w+1; r++) {
		for (int c=0; c<2*w+1; c++) {
			x1[r*(2*w+1)+c] = -w+c;
			// cout << x1[r*(2*w+1)+c] << " " ;
		}
		// cout << endl;		
	}
	// cout << "y1: " << endl;
	for (int r=0; r<2*w+1; r++) {
		for (int c=0; c<2*w+1; c++) {
			y1[r*(2*w+1)+c] = -w+r;
			// cout << y1[r*(2*w+1)+c] << " " ;
		}
		// cout << endl;		
	}
	// cout <<"mg1"<< endl;
	float* gx1 = new float[(int)pow(2*w+1,2)];
	g1x(gx1,x1,y1,std1,w);
	// for (int r=0; r<2*w+1; r++) {
	// 	for (int c=0; c<2*w+1; c++) {
	// 		cout << gx1[r*(2*w+1)+c] << " " ;
	// 	}
	// 	cout << endl;
	// }
	float* gy1 = new float[(int)pow(2*w+1,2)];
	g1y(gy1,x1,y1,std1,w);
	float* gimx = new float[width*height];
	filter(gimx,gx1,I,width,height,w);
    writePGM((uchar*)gimx,width,height,"gimx.pgm");
	for (int r=0; r<height; r++) {
		for (int c=0; c<width; c++) {
			cout << gimx[r*width+c] << " " ;
		}
		cout << endl;
	}
	float* gimy = new float[width*height];
	filter(gimy,gy1,I,width,height,w);
	float* mg1 = new float[width*height];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			mg1[i*width+j] = sqrt(pow(gimx[i*width+j],2)+pow(gimy[i*width+j],2));
			// cout << mg1[i*height+j] << " ";
		}
		// cout << endl;
	}

	delete []x1;
	delete []y1;
	delete []gx1;
	delete []gy1;
	delete []gimx;
	delete []gimy;

	int w2=  (2*ceil(2* std2))+1;
	int* x2 = new int[(int)pow(2*w2+1,2)];
	int* y2 = new int[(int)pow(2*w2+1,2)];
	// cout << "x2: " << endl;
	for (int r=0; r<2*w2+1; r++) {
		for (int c=0; c<2*w2+1; c++) {
			x2[r*(2*w2+1)+c] = -w2+c;
			// cout << x2[r*(2*w2+1)+c] << " " ;
		}
		// cout << endl;		
	}
	// cout << "y2: " << endl;
	for (int r=0; r<2*w2+1; r++) {
		for (int c=0; c<2*w2+1; c++) {
			y2[r*(2*w2+1)+c] = -w2+r;
			// cout << y2[r*(2*w2+1)+c] << " " ;
		}
		// cout << endl;		
	}

	// cout << "mg2: " <<endl;
	float* gx2 = new float[(int)pow(2*w2+1,2)];
	g1x(gx2,x2,y2,std2,w2);
	float* gy2 = new float[(int)pow(2*w2+1,2)];
	g1y(gy2,x2,y2,std2,w2);
	float* gimx2 = new float[width*height];
	filter(gimx2,gx2,I,width,height,w2);
	float* gimy2 = new float[width*height];
	filter(gimy2,gy2,I,width,height,w2);
	float* mg2 = new float[width*height];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			mg2[i*width+j] = sqrt(pow(gimx2[i*width+j],2)+pow(gimy2[i*width+j],2));
			// cout << mg2[i*height+j] << " ";
		}
		// cout << endl;
	}		

	delete []x2;
	delete []y2;
	delete []gx2;
	delete []gy2;
	delete []gimx2;
	delete []gimy2;	

	cout << "gRatio" << endl;
	float* gRatio = new float[width*height];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			gRatio[i*width+j] = mg1[i*width+j] / mg2[i*width+j];
			// cout << gRatio[i*width+j] << " " ;
		}
		// cout << endl;
	}
    writePGM((uchar*)gRatio,width,height,"gRatio.pgm");


	delete []mg1;
	delete []mg2;
	// cout << "sparse" << endl;
	float* sparse = new float[width*height];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			if (edge[i*width+j] != 0 ) {
				if (gRatio[i*width+j]>1.01 )//&& gRatio[i*height+j]<= std2/std1)
					sparse[i*width+j] = sqrt(pow(gRatio[i*width+j],2)*pow(std1,2)-pow(std2,2))/(1.0-pow(gRatio[i*width+j],2));
        // sparseDMap(idx(jj))=sqrt((gRatio(idx(jj)).^2*std1^2-std2^2)/(1-gRatio(idx(jj)).^2));

				else
					sparse[i*width+j] = 0;
			}
			else
				sparse[i*width+j] = 0;
			// cout << sparse[i*height+j] << " ";
		}
		// cout << endl;
	}	
	delete []gRatio;

	return (uchar*)sparse;

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
			gim[i*width+j] = sum/count;
			// cout << "r: " << sum << " " << count << " " << sum/count <<" " << 
			// cout << gim[i*width+j] << " ";
		}
		// cout << endl;
	}
}