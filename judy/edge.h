#include <iostream>
#include <math.h>

using namespace std;
typedef unsigned char uchar;

#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0
#define NOEDGE 0
#define POSSIBLE_EDGE 128
#define EDGE 255

void canny(uchar *image, int rows, int cols, float sigma,
         float tlow, float thigh, uchar **edge, char *fname);
void gaussian_smooth(uchar *image, int rows, int cols, float sigma, short int **smoothedim);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void derrivative_x_y(short int *smoothedim, int rows, int cols, short int **delta_x, short int **delta_y);
void radian_direction(short int *delta_x, short int *delta_y, int rows,
    int cols, float **dir_radians, int xdirtag, int ydirtag);
void follow_edges(uchar *edgemapptr, short *edgemagptr, short lowval, int cols);
void apply_hysteresis(short int *mag, uchar *nms, int rows, int cols,
	float tlow, float thigh, uchar *edge);
void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, uchar *result) ;
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols, short int **magnitude);
double angle_radians(double x, double y);