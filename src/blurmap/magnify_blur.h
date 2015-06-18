void generateGaussianTable(float* gaussian_table, const float sigma, const int length)
{
	const float denominator_inverse = -1.0f / (2.0f * sigma * sigma);
	for (int i = 0; i < length; ++i) {
		gaussian_table[i] = exp(i*i*denominator_inverse);
	}
}



//construct d2r map
size_t* d2r_table = new size_t[256];
int r_arr[4] = {0, width/108, width/48, width/30};
int sigma_arr[4] = {0, max(1, r_arr[1]/3), max(1, r_arr[2]/3), max(1, r_arr[3]/3) };
for(int i = 0; i < 20; ++i) {
	d2r_table[i] = r_arr[0];
}
for(int i = 20; i < 60; ++i) {
	d2r_table[i] = r_arr[1];
}
for(int i = 60; i < 140; ++i) {
	d2r_table[i] = r_arr[2];
}
for(int i = 140; i <= 255; ++i) {
	d2r_table[i] = r_arr[3];
}

//assign r to each pixel
size_t* r_map = new size_t[numPixel];
for(int i = 0; i < numPixel; ++i) {
	r_map[i] = d2r_table[blurmap[i]];
}

//construct group map
size_t* grp_table = new size_t[256];
for(int i = 0; i < 20; ++i) {
	grp_table[i] = 0;
}
for(int i = 20; i < 60; ++i) {
	grp_table[i] = 1;
}
for(int i = 60; i < 140; ++i) {
	grp_table[i] = 2;
}
for(int i = 140; i <= 255; ++i) {
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


//depth dependence gaussian filter
int r, sigma;
float* ftdIm = new float[numPixel];
for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const uchar *base_in = &Im[width*y+x];
			float *base_out = &ftdIm[width*y+x];
			float weight_sum = 0.0f;
			float weight_pixel_sum = 0.0f;
			r = r_arr[grp[width*y+x]];
			sigma = sigma_arr[grp[width*y+x]];
			if(r==0) {
				base_out[0] = base_in[0];
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
					weight_pixel_sum += weight * base_in[dy*width+dx];
				}
			}
			base_out[0] = (int(weight_pixel_sum/weight_sum + 0.5f));
		}
}