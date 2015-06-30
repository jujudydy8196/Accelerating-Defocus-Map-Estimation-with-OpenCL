
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <ctime>
#include <iostream>
#include <fstream>
#include "fileIO.h"
#include "edge.h"
#include "defocus.h"

using namespace std;
//using namespace cv;

typedef unsigned char uchar;

void imageGray( float* image, float* gray, int size );

void imageUchar2Float( uchar* uImage, float* fImage, const int size );
void imageFloat2Uchar( float* fImage, uchar* uImage, const int size );

int main(int argc, char** argv) {
	if(argc !=3) {
	    cout << "Usage: sparse_map <original image> <1:blur after sparse/ 2:no blur>" << endl;
	    return -1;
	}

	  uchar* I_ori_uchar = NULL;
	  uchar* I_edge_uchar = NULL;
    float* I_sparse = NULL;
    float* I_ori = NULL;
    float* I_gray = NULL;
    float* I_edge = NULL;
    float* I_edge_mat = NULL;
    int width, height, numPixel, n;

    sizePGM(width, height, argv[1]);
    numPixel = width*height;
    n = numPixel * 3;

    I_sparse = new float[numPixel];
    I_gray = new float[numPixel];
    I_edge = new float[numPixel];
    I_edge_mat = new float[numPixel];
    I_ori = new float[n];
    I_ori_uchar = new uchar[n];
    I_edge_uchar = new uchar[numPixel];

		clock_t start, stop;
		ofstream timelog;
		timelog.open("time.txt",ios::app);

    readPPM(I_ori_uchar, argv[1]);
    imageUchar2Float( I_ori_uchar, I_ori, n );

    imageGray( I_ori, I_gray, numPixel );
		//Mat I_cv = imread(argv[1]);
		//Mat I_gray_cv,I_edge_cv;
		//cvtColor(I_cv,I_gray_cv,CV_BGR2GRAY);
		//

		start = clock();
		canny(I_gray, height, width, 1, 0.5, 0.8, &I_edge, "test");
		stop = clock();

		imageFloat2Uchar(I_edge,I_edge_uchar,numPixel);
		
		cout << "canny detection time: " << double(stop - start) / CLOCKS_PER_SEC << endl;
		timelog << "canny detection time: " << double(stop - start) / CLOCKS_PER_SEC << endl;

		//for (int i=0; i<numPixel; i++)
				//if (I_edge[i]!=0) cout << I_edge[i] << " " ;

		//cout<< endl;

		//readPPM(I_edge_uchar, "testedge.ppm");
		//for (int i=0; i<numPixel; i++)
				//if (I_edge_uchar[i]!=0) cout << (float)I_edge_uchar[i] << " " ;

		//cout<< endl;
		//imageUchar2Float( I_edge_uchar, I_edge_mat, n );

		//for (int i=0; i<numPixel; i++)
				//if (I_edge_mat[i]!=0) cout << I_edge_mat[i] << " " ;
		
		//cout<< endl;
    //Canny(I_gray_cv,I_edge_cv,0.5,0.8);
		//imwrite("edge_cv.pgm",I_edge_cv);

		//readPGM(I_edge_uchar, argv[1]);
		//imageUchar2Float( I_edge_uchar, I_edge, numPixel );

		start = clock();
    defocusEstimation(I_ori,I_gray, I_edge, I_sparse, 1.0, 0.001, 3, width, height,atoi(argv[2])) ;
		stop = clock();
		cout << "defocus Estimation time: " << double(stop - start) / CLOCKS_PER_SEC << endl;
		timelog << "defocus Estimation time: " << double(stop - start) / CLOCKS_PER_SEC << endl;


}

void imageGray( float* image, float* gray, int size ){
    for( size_t i = 0; i < size; ++i ){
        gray[i] = 0.2126*image[3*i] + 0.7152*image[3*i+1] + 0.0722*image[3*i+2];
    }
}
void imageFloat2Uchar( float* fImage, uchar* uImage, const int size )
{
    for( size_t i = 0; i < size; ++i ){
        uImage[i] = uchar(fImage[i] * 255.0);
    }
}
void imageUchar2Float( uchar* uImage, float* fImage, const int size )
{
    for( size_t i = 0; i < size; ++i ){
        fImage[i] = uImage[i] / 255.0f;
    }
}
