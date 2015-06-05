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
	imageInfo( gimx, width*height );
	writeDiff( gimx, width, height, "gimx.pgm" );	
    // writePGM((uchar*)gimx,width,height,"gimx.pgm");
	// for (int r=0; r<height; r++) {
	// 	for (int c=0; c<width; c++) {
	// 		cout << gimx[r*width+c] << " " ;
	// 	}
	// 	cout << endl;
	// }
	float* gimy = new float[width*height];
	filter(gimy,gy1,I,width,height,w);
	imageInfo( gimy, width*height );
	writeDiff( gimy, width, height, "gimy.pgm" );	
	float* mg1 = new float[width*height];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			mg1[i*width+j] = sqrt(pow(gimx[i*width+j],2)+pow(gimy[i*width+j],2));
			// cout << mg1[i*height+j] << " ";
		}
		// cout << endl;
	}
	imageInfo( mg1, width*height );
	write( mg1, width, height, "mg1.pgm" );	
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
	imageInfo( gimx2, width*height );
	writeDiff( gimx2, width, height, "gimx2.pgm" );		
	float* gimy2 = new float[width*height];
	filter(gimy2,gy2,I,width,height,w2);
	imageInfo( gimy2, width*height );
	writeDiff( gimy2, width, height, "gimy2.pgm" );		
	float* mg2 = new float[width*height];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			mg2[i*width+j] = sqrt(pow(gimx2[i*width+j],2)+pow(gimy2[i*width+j],2));
			// cout << mg2[i*height+j] << " ";
		}
		// cout << endl;
	}		
	imageInfo( mg2, width*height );
	write( mg2, width, height, "mg2.pgm" );	

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
			if (mg2[i*width+j])
				gRatio[i*width+j] = mg1[i*width+j] / mg2[i*width+j];
			else {
				// cout <<"mg2=0" << endl;
				gRatio[i*width+j] = 0;
			}
			// cout << gRatio[i*width+j] << " " ;
		}
		// cout << endl;
	}
    // writePGM((uchar*)gRatio,width,height,"gRatio.pgm");
	imageInfo( gRatio, width*height );
	write( gRatio, width, height, "gRatio.pgm" );	

	delete []mg1;
	delete []mg2;
	// cout << "sparse" << endl;
	float* sparse = new float[width*height];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			if (edge[i*width+j] != 0 ) {
				// cout << "1: " <<  1.0-pow(gRatio[i*width+j],2) << endl;

				if (gRatio[i*width+j]>1.01){ // && (1.0-pow(gRatio[i*width+j],2))>0 ) {
					sparse[i*width+j] = sqrt((pow(gRatio[i*width+j],2)*pow(std1,2)-pow(std2,2))/(1.0-pow(gRatio[i*width+j],2)));
					// cout << "1: " <<  1.0-pow(gRatio[i*width+j],2) << endl;
					// cout << "2: " << pow(gRatio[i*width+j],2)*pow(std1,2)-pow(std2,2) << endl;
					// cout << "3: " << sqrt(pow(gRatio[i*width+j],2)*pow(std1,2)-pow(std2,2))/(1.0-pow(gRatio[i*width+j],2)) << endl;
				}
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
	sparseScale(sparse,maxBlur,height*width);
	// imageInfo( sparse, width*height );
	write( sparse, width, height, "sparse.pgm" );	
	delete []gRatio;

	return (uchar*)sparse;

}
void sparseScale(float* I, int maxBlur, size_t size) {
	float max = I[0];
	for( size_t i = 1; i < size; ++i ){
		if( I[i] > maxBlur )
			I[i] = maxBlur;
		I[i] = I[i] / maxBlur * 255.0;
	}
	// cout << "sparsemin: " << double(min) << ", sparsemax: " << double(max) << endl;
	// for (size_t i = 0 ; i<size; ++i) {
		// I[i] = I[i] / max * maxBlur;
}
void g1x(float* g, int* x, int* y, float std, int w) {
	float squareStd = pow(std,2);
	for (int i=0; i<pow(2*w+1,2); i++)
		g[i] = -(x[i]/(2*PI*pow(squareStd,2)))* exp(-(pow(x[i],2)+pow(y[i],2))/(2*squareStd));
// g = -(x./(2*pi*s1sq.^2)) .* exp(-(x.^2 + y.^2)./(2*s1sq)); 
}

void g1y(float* g, int* x, int* y, float std, int w) {
	float squareStd = pow(std,2);
	for (int i=0; i<pow(2*w+1,2); i++)
		g[i] = -(y[i]/(2*PI*pow(squareStd,2)))* exp(-(pow(x[i],2)+pow(y[i],2))/(2*squareStd));
// g = -(y./(2*pi*s1sq.^2)) .* exp(-(x.^2 + y.^2)./(2*s1sq)); 

}


void filter(float* gim, float* g , uchar* I, int width, int height, int w) {
	// cout << " w: " << w << endl;
	for( size_t i = 0; i < width * height; ++i ){
		gim[i] = 0;
	}

	for(int i=w; i<height-w; i++) {
		for( int j=w; j<width-w; j++) {
			// int count=0;
			float sum=0;
			for(int x=0; x<2*w+1; x++) {
				for (int y=0; y<2*w+1; y++) {
					if ((i-w+y)<0 || (i-w+y)>=height)
						continue;
					else if((j-w+x)<0 || (j-w+x)>=width)
						continue;
					else {
						// sum +=( (float)I[(i-w+y)*width+(j-w+x)] * g[y*(2*w+1)+x]);
						sum +=( (float)I[(i-w+y)*width+(j-w+x)] * g[(2*w-y)*(2*w+1)+(2*w-x)]);
						// cout << "x: " << x << " y: " << y << " " << "i: " << i << " j: " << j ;
						// cout << " I: " <<(float)I[(i-w+y)*width+(j-w+x)]/255.0 << " g: " << g[y*(2*w+1)+x]<< " " << endl;
						// cout << " count: " << count << endl;
						// count++;
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


template <class T>
void imageInfo( T* I, size_t size )
{
	T max = I[0], min = I[0];
	// size_t count[17] = {};
	for( size_t i = 1; i < size; ++i ){
		if( max < I[i] ) max = I[i];
		if( min > I[i] ) min = I[i];

		// if( size_t( I[i]/10+8 ) > 16 ) cout << I[i] << endl;
		// else ++count[size_t( I[i]/10+8 )];
	}
	cout << "min: " << double(min) << ", max: " << double(max) << endl;
	// for(size_t i = 0; i < 17; ++i)
		// cout << count[i] << ' ';
	cout << endl;
}

void writeDiff( const float* I, size_t w, size_t h, char* str )
{
	size_t size = w * h;
	uchar* out = new uchar[size];
	for (size_t i = 0; i < size; ++i)
	{
		out[i] = uchar((I[i]+128));//uchar( I[i]/270*128 + 128 );
	}
    writePGM(out,w,h,str);
    delete [] out;
}

void write( const float* I, size_t w, size_t h, char* str )
{
	size_t size = w * h;
	uchar* out = new uchar[size];
	for (size_t i = 0; i < size; ++i)
	{
		out[i] = uchar((I[i]));//uchar( I[i]/270*128 + 128 );
	}
    writePGM(out,w,h,str);
    delete [] out;
}
