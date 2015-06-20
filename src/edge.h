#include <iostream>
#include <math.h>

using namespace std;
typedef unsigned char uchar;

#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0
#define NOEDGE 0
#define POSSIBLE_EDGE 128.0
#define EDGE 255.0

void canny(float *image, int rows, int cols, float sigma,
         float tlow, float thigh, float **edge, char *fname);
void gaussian_smooth(float *image, int rows, int cols, float sigma, short int **smoothedim);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void derrivative_x_y(short int *smoothedim, int rows, int cols, short int **delta_x, short int **delta_y);
void radian_direction(short int *delta_x, short int *delta_y, int rows,
    int cols, float **dir_radians, int xdirtag, int ydirtag);
void follow_edges(float *edgemapptr, short *edgemagptr, short lowval, int cols);
void apply_hysteresis(short int *mag, float *nms, int rows, int cols,
	float tlow, float thigh, float *edge);
void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, float *result) ;
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols, short int **magnitude);
double angle_radians(double x, double y);