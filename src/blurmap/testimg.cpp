#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include "fileIO.h"

using namespace std;

void generateGaussianTable(float* gaussian_table, const float sigma, const int length)
{
	const float denominator_inverse = -1.0f / (2.0f * sigma * sigma);
	for (int i = 0; i < length; ++i) {
		gaussian_table[i] = exp(i*i*denominator_inverse);
	}
}

void imageFloat2Uchar( float* fImage, uchar* uImage, const int size )
{
    for( size_t i = 0; i < size; ++i ){
        uImage[i] = uchar(fImage[i] * 255.0);
    }
}

int main(int argc, char** argv ) {
	int width, height, numPixel;
	uchar* blurmap;
	uchar* Im;
	uchar* ftdIm;
	sizePGM(width, height, argv[1]);
	cout<<"width "<<width<<" height "<<height;
	numPixel = width*height;
	blurmap = new uchar[numPixel];
	Im = new uchar[3*numPixel];
	ftdIm = new uchar[3*numPixel];
	readPPM(Im, argv[1]);
	readPGM(blurmap, argv[2]);
	int r_arr[4] = {0, width/160, width/80, width/60};
	int sigma_arr[4] = {0, max(1, r_arr[1]/3), max(1, r_arr[2]/3), max(1, r_arr[3]/3) };

	cout<<r_arr[0]<<" "<<r_arr[1]<<" "<<r_arr[2]<<" "<<r_arr[3]<<endl;
	//construct group map
	size_t* grp_table = new size_t[256];
	for(int i = 0; i < 145; ++i) {
		grp_table[i] = 0;
	}
	for(int i = 145; i < 170; ++i) {
		grp_table[i] = 1;
	}
	for(int i = 170; i < 200; ++i) {
		grp_table[i] = 2;
	}
	for(int i = 200; i <= 255; ++i) {
		grp_table[i] = 3;
	}

	//assign group to each pixel
	size_t* grp = new size_t[numPixel];
	for(int i = 0; i < numPixel; ++i) {
		grp[i] = grp_table[blurmap[i]];
	}

	//construct gaussian table
	float** gaussian_table_arr = new float*[4];
	for(int i = 0; i < 4; ++i) {
		gaussian_table_arr[i] = new float[r_arr[i]+1];
		generateGaussianTable(gaussian_table_arr[i], sigma_arr[i], r_arr[i]+1);
	}
/*	
	char numstr[5];
	for(int i = 0; i < 4; ++i) {
		ostringstream buffer;
		buffer << "gaussian" << i << ".txt.";
		ofstream outfile(buffer.str().c_str());
		for(int j = 0; j < r_arr[i]+1; ++j) {
			outfile<<gaussian_table_arr[i][j]<<" ";
		}
		outfile.close();
	}
*/

	//depth dependence gaussian filter
	//ofstream outfile("checkDXY.txt");
	int r, sigma, hh;
	hh = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const uchar *base_in = &Im[3*(width*y+x)];
			uchar *base_out = &ftdIm[3*(width*y+x)];
			float weight_sum = 0.0f;
			float weight_pixel_sum_r = 0.0f;
			float weight_pixel_sum_g = 0.0f;
			float weight_pixel_sum_b = 0.0f;
			r = r_arr[grp[width*y+x]];
			sigma = sigma_arr[grp[width*y+x]];
			if(r==0) {
				base_out[0] = base_in[0];
				base_out[1] = base_in[1];
				base_out[2] = base_in[2];
			//	cout<<base_out[0]<<" "<<base_out[1]<<" "<<base_out[2]<<endl;
				continue;
			}
			for (int dy = -r; dy <= r; dy++) {
				if( ((y+dy) > height) || ((y+dy) < 0) ) continue;
				for (int dx = -r; dx <= r; dx++) {
					if( ((x+dx) > width) || ((x+dx) < 0) ) continue;
					int range_xdiff = abs(dx);
					int range_ydiff = abs(dy);
					float weight =
						 gaussian_table_arr[grp[width*y+x]][range_xdiff]
						*gaussian_table_arr[grp[width*y+x]][range_ydiff];
					weight_sum += weight;
					weight_pixel_sum_r += weight * base_in[3*(dy*width+dx)  ];
					weight_pixel_sum_g += weight * base_in[3*(dy*width+dx)+1];
					weight_pixel_sum_b += weight * base_in[3*(dy*width+dx)+2];
				}
			}
			float out_r = (int(weight_pixel_sum_r/weight_sum + 0.5f));
			float out_g = (int(weight_pixel_sum_g/weight_sum + 0.5f));
			float out_b = (int(weight_pixel_sum_b/weight_sum + 0.5f));
			if(out_r > 255) out_r = 255;
			else if (out_r < 0) out_r = 0;
			else;
			if(out_g > 255) out_g = 255;
			else if (out_g < 0) out_g = 0;
			else;
			if(out_b > 255) out_b = 255;
			else if (out_b < 0) out_b = 0;
			else;
			base_out[0] = out_r;
			base_out[1] = out_g;
			base_out[2] = out_b;
		}
		hh++;
	}
	ofstream outfile("checkResult.txt");
	for(int y = 0; y < height; ++y) {
		outfile<<" y "<<y<<endl;
		for(int x = 0; x < width; ++x) {
			outfile<<(int)ftdIm[3*(y*width+x)  ]<<" ";
			outfile<<(int)ftdIm[3*(y*width+x)+1]<<" ";
			outfile<<(int)ftdIm[3*(y*width+x)+2]<<" ";
		}
	}
	outfile.close();
	cout<<"h "<<hh<<endl;
	writePPM(ftdIm, width, height, "blurredImage.pgm");	
	delete blurmap;
	delete Im;
	delete ftdIm;
	return 0;

}
