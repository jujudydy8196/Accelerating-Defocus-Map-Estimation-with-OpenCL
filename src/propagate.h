#include "vec.h"

void propagate( uchar*, uchar*, size_t, size_t, float, size_t, Vec<uchar>& );
void constructH( const uchar*, Vec<uchar>& H, const size_t);
void constructEstimate( uchar*, Vec<float>& );
void conjgrad( const Vec<uchar>&, float*,const LaplaMat*, float*, const Vec<float>&, Vec<float>&, size_t, size_t, float );
void vecFloat2uchar( const Vec<float>&, Vec<uchar>& );

void HFilter(float* , const float* , const Vec<uchar>& , const size_t);
void lambda_LFilter(float*, const uchar*, const float*, const int, const int, const float, const int );
void getAp(float*, const float*, const float*, const int );

void propagate( uchar* image, uchar* estimatedBlur, size_t w, size_t h, float lambda, size_t radius, Vec<uchar>& result )
{
    size_t size = w * h;
    Vec<float> estimate( size ), x( size );
	Vec<uchar> H( size );
    float* Hp = new float[size];
    float* Lp = new float[size];
	LaplaMat* LM = new LaplaMat(image, w, h, radius);
    constructEstimate( estimatedBlur, estimate );
	constructH( estimatedBlur, H, size);
    conjgrad( H, Hp, LM, Lp, estimate, x, w, h, lambda );
    vecFloat2uchar( x, result );

    delete [] Hp;
    delete [] Lp;
	delete LM;
}

void constructH( const uchar* estimatedBlur, Vec<uchar>& H, const size_t numPixel) {
    for(size_t i = 0; i < numPixel; ++i) {
            if((int)estimatedBlur[i] != 0) { H[i] = 1;}
            else H[i] = 0;
    }
}

void constructEstimate( uchar* estimatedBlur, Vec<float>& estimate )
{
    size_t size = estimate.getSize();
    for( size_t i = 0; i < size; ++i ){
        estimate[i] = float( estimatedBlur[i] );
    }  
}

void conjgrad(const Vec<uchar>& H, float* Hp, const LaplaMat* LM, float* Lp, const Vec<float>& estimate, Vec<float>& x, size_t w, size_t h, float lambda )
{
    cout << "in conjgrad\n";
    size_t size = estimate.getSize();
    Vec<float> r( estimate ), p( estimate ), Ap( size );
    float rsold = Vec<float>::dot( r, r ), alpha = 0.0, rsnew = 0.0;
	
    for( size_t i = 0; i < 1000000; ++i ){
        // Ap = A * p
        cout << i << ' ' << rsold << endl;
        HFilter( Hp, p.getPtr(), H, size );
        // lambda_LFilter( L, image, p.getPtr(), h, w, lambda, radius );
		LM->run(Lp, p.getPtr(), lambda);
        getAp( Ap.getPtr(), Hp, Lp, size );
        alpha = rsold / Vec<float>::dot( p, Ap );
        Vec<float>::add( x, x, p, 1, alpha );
        Vec<float>::add( r, r, Ap, 1, -alpha );
        rsnew = Vec<float>::dot( r, r );
        if( rsnew < 1e-20 ) break;
        Vec<float>::add( p, r, p, 1, rsnew/rsold );
        rsold = rsnew;
    }
}

void vecFloat2uchar( const Vec<float>& F, Vec<uchar>& U )
{
    size_t size = F.getSize();
    for( size_t i = 0; i < size; ++i ){
        U[i] = float( F[i] );
    }
}

void HFilter(float* Hp, const float* p, const Vec<uchar>& H, const size_t numPixel) {
	for(size_t i = 0; i < numPixel; ++i) {
		if(H[i]) Hp[i] = p[i];
		else Hp[i] = 0;
	}
}

void lambda_LFilter(float* Lp, const uchar* I_ori, const float* p, const int height, const int width, const float lambda, const int r) {
    int idx, numWinPixel;
    numWinPixel = r*r;
    float* tmpI = new float[height*width];
    //W*I
    guided_filter gf(I_ori, width, height, r, 0.00001);
    gf.run(p, tmpI);
    
    for(int y = 0; y < height; ++y) {
        for(int x = 0; x < width; ++x) {
            idx = y*width + x;
            //I-W*I
            tmpI[idx] = (float)I_ori[idx] - tmpI[idx];
            //L = |w|*(I-W)     //L = lamda*L
            Lp[idx] = (float)lambda * (float)numWinPixel * tmpI[idx];
        }
    }
}

//Ap = (H + lamda_L)p = Hp + lamda_L*p
void getAp(float* Ap, const float* Hp, const float* Lp, const int numPixel) {
    for(int i = 0; i < numPixel; ++i) Ap[i] = (float)Hp[i] + Lp[i]; 
}