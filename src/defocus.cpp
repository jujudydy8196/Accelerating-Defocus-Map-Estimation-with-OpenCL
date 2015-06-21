#include "defocus.h"
#include "fileIO.h"
#include "guidedfilter.h"

void defocusEstimation(float* I_rgb, float* I, float* edge, float* out, float std, float lamda, float maxBlur, int width, int height, int blur) {
	// std :the standard devitation reblur gaussian1, typically std=[0.8:1]

	//write( edge, width, height, "1edge.pgm" );
	float std1= std;
	float std2= std1 * 1.5;
	int w=  (2*ceil(2* std1))+1;
	int* x1 = new int[(int)pow(2*w+1,2)];
	int* y1 = new int[(int)pow(2*w+1,2)];
	for (int r=0; r<2*w+1; r++) {
		for (int c=0; c<2*w+1; c++) {
			x1[r*(2*w+1)+c] = -w+c;
		}
	}
	for (int r=0; r<2*w+1; r++) {
		for (int c=0; c<2*w+1; c++) {
			y1[r*(2*w+1)+c] = -w+r;
		}
	}
	float* gx1 = new float[(int)pow(2*w+1,2)];
	g1x(gx1,x1,y1,std1,w);
	float* gy1 = new float[(int)pow(2*w+1,2)];
	g1y(gy1,x1,y1,std1,w);
	float* gimx = new float[width*height];
	filter(gimx,gx1,I,width,height,w);
	//imageInfo( gimx, width*height );
	// writeDiff( gimx, width, height, "gimx.pgm" );	
    float* gimy = new float[width*height];
	filter(gimy,gy1,I,width,height,w);
	//imageInfo( gimy, width*height );
	// writeDiff( gimy, width, height, "gimy.pgm" );	
	float* mg1 = new float[width*height];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			mg1[i*width+j] = sqrt(pow(gimx[i*width+j],2)+pow(gimy[i*width+j],2));
		}
	}
	//imageInfo( mg1, width*height );
	// write( mg1, width, height, "mg1.pgm" );	
	delete []x1;
	delete []y1;
	delete []gx1;
	delete []gy1;
	delete []gimx;
	delete []gimy;

	int w2=  (2*ceil(2* std2))+1;
	int* x2 = new int[(int)pow(2*w2+1,2)];
	int* y2 = new int[(int)pow(2*w2+1,2)];
	for (int r=0; r<2*w2+1; r++) {
		for (int c=0; c<2*w2+1; c++) {
			x2[r*(2*w2+1)+c] = -w2+c;
		}
	}
	for (int r=0; r<2*w2+1; r++) {
		for (int c=0; c<2*w2+1; c++) {
			y2[r*(2*w2+1)+c] = -w2+r;
		}
	}

	float* gx2 = new float[(int)pow(2*w2+1,2)];
	g1x(gx2,x2,y2,std2,w2);	
	float* gy2 = new float[(int)pow(2*w2+1,2)];
	g1y(gy2,x2,y2,std2,w2);
	float* gimx2 = new float[width*height];
	filter(gimx2,gx2,I,width,height,w2);
	//imageInfo( gimx2, width*height );
	// writeDiff( gimx2, width, height, "gimx2.pgm" );		
	float* gimy2 = new float[width*height];
	filter(gimy2,gy2,I,width,height,w2);
	//imageInfo( gimy2, width*height );
	// writeDiff( gimy2, width, height, "gimy2.pgm" );		
	float* mg2 = new float[width*height];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			mg2[i*width+j] = sqrt(pow(gimx2[i*width+j],2)+pow(gimy2[i*width+j],2));
		}
	}		
	//imageInfo( mg2, width*height );
	// write( mg2, width, height, "mg2.pgm" );	

	delete []x2;
	delete []y2;
	delete []gx2;
	delete []gy2;
	delete []gimx2;
	delete []gimy2;	

	// cout << "gRatio" << endl;
	float* gRatio = new float[width*height];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			if (mg2[i*width+j])
				gRatio[i*width+j] = mg1[i*width+j] / mg2[i*width+j];
			else {
				gRatio[i*width+j] = 0;
			}
		}
	}
	//imageInfo( gRatio, width*height );
	// write( gRatio, width, height, "gRatio.pgm" );	

	delete []mg1;
	delete []mg2;
	// float* sparse = new float[width*height];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			if (edge[i*width+j] != 0 ) {

				if (gRatio[i*width+j]>1.01  && (pow(gRatio[i*width+j],2)*pow(std1,2)-pow(std2,2))/(1.0-pow(gRatio[i*width+j],2))>0 ) {
					out[i*width+j] = sqrt((pow(gRatio[i*width+j],2)*pow(std1,2)-pow(std2,2))/(1.0-pow(gRatio[i*width+j],2)));
			}
				else
					out[i*width+j] = 0;
			}
			else
				out[i*width+j] = 0;
			// cout << out[i*width+j] << " " << (pow(gRatio[i*width+j],2)*pow(std1,2)-pow(std2,2))/(1.0-pow(gRatio[i*width+j],2)) << endl;
		}
	}	
	sparseScale(out,maxBlur,height*width);
	//imageInfo( out, width*height );

  if (blur == 2)
		write( out, width, height, "sparse.pgm" );
	else if ( blur == 1) {
		write( out, width, height, "no_blur_sparse.pgm" );
    guided_filter gf (I_rgb, width, height, 2, 0.00001);
    gf.run(out,out);
		write( out, width, height, "sparse.pgm" );
	}

	// for(size_t i = 0; i < height * width; ++i ){
	// 	cout << out[i] <<" ";
	// }

	delete [] gRatio;
	// delete [] sparse;
}
void sparseScale(float* I, int maxBlur, size_t size) {
	for( size_t i = 1; i < size; ++i ){
		if( I[i] > maxBlur )
			I[i] = maxBlur;
		I[i] = I[i] / maxBlur ;//* 255.0;
	}
}
void g1x(float* g, int* x, int* y, float std, int w) {
	float squareStd = pow(std,2);
	for (int i=0; i<pow(2*w+1,2); i++)
		g[i] = -(x[i]/(2*PI*pow(squareStd,2)))* exp(-(pow(x[i],2)+pow(y[i],2))/(2*squareStd));
}

void g1y(float* g, int* x, int* y, float std, int w) {
	float squareStd = pow(std,2);
	for (int i=0; i<pow(2*w+1,2); i++)
		g[i] = -(y[i]/(2*PI*pow(squareStd,2)))* exp(-(pow(x[i],2)+pow(y[i],2))/(2*squareStd));
}


void filter(float* gim, float* g , float* I, int width, int height, int w) {
	for( size_t i = 0; i < width * height; ++i ){
		gim[i] = 0;
	}

	for(int i=w; i<height-w; i++) {
		for( int j=w; j<width-w; j++) {
			float sum=0;
			for(int x=0; x<2*w+1; x++) {
				for (int y=0; y<2*w+1; y++) {
					if ((i-w+y)<0 || (i-w+y)>=height)
						continue;
					else if((j-w+x)<0 || (j-w+x)>=width)
						continue;
					else {
						sum +=( (float)I[(i-w+y)*width+(j-w+x)] * g[(2*w-y)*(2*w+1)+(2*w-x)]);
					}
				}
			}
			gim[i*width+j] = sum;//count;
		}
	}
}


template <class T>
void imageInfo( T* I, size_t size )
{
	T max = I[0], min = I[0];
	for( size_t i = 1; i < size; ++i ){
		if( max < I[i] ) max = I[i];
		if( min > I[i] ) min = I[i];
	}
	cout << "min: " << double(min) << ", max: " << double(max) << endl;
	cout << endl;
}

void writeDiff( const float* I, size_t w, size_t h, char* str )
{
	size_t size = w * h;
	uchar* out = new uchar[size];
	for (size_t i = 0; i < size; ++i)
	{
		out[i] = uchar((I[i]*255+128));//uchar( I[i]/270*128 + 128 );
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
		// cout << I[i] << endl;
		out[i] = uchar((I[i]*255));//uchar( I[i]/270*128 + 128 );
	}
    writePGM(out,w,h,str);
    delete [] out;
}
