#include "propagate.h"

void propagate( const float* image, const float* estimatedBlur, const size_t w, const size_t h, const float lambda, const size_t radius, Vec<float>& result )
{
    size_t size = w * h;

    for(size_t i = 0; i < size; ++i){
        if( estimatedBlur[i] < 0 || estimatedBlur[i] > 1 ) cout << estimatedBlur[i] << ' ';
    }
    cout << endl;

    Vec<float> estimate( size );//, x( size );
    Vec<float> H( size );
    float* Hp = new float[size];
    float* Lp = new float[size];
    LaplaMat* LM = new LaplaMat(image, w, h, radius);
    constructEstimate( estimatedBlur, estimate );
    constructH( estimatedBlur, H, size);
    conjgrad( H, Hp, LM, Lp, estimate, result, w, h, lambda );
    // vecFloat2uchar( x, result );

    delete [] Hp;
    delete [] Lp;
    delete LM;
}

void constructH( const float* estimatedBlur, Vec<float>& H, const size_t numPixel) {
    for(size_t i = 0; i < numPixel; ++i) {
            if(estimatedBlur[i] != 0) { H[i] = 1;}
            else H[i] = 0;
    }
}

void constructEstimate( const float* estimatedBlur, Vec<float>& estimate )
{
    size_t size = estimate.getSize();
    for( size_t i = 0; i < size; ++i ){
        //estimate[i] = float( estimatedBlur[i] )/255;
        estimate[i] = estimatedBlur[i];
    //  of<<estimate[i]<<" ";
    }  
}

void printEstimate( const Vec<float>& estimate, const size_t size, ofstream& of)
{
    for( size_t i = 0; i < size; ++i ){
        of<<estimate[i]<<" ";
    }  
    
}

void printP( const Vec<float>& p, const size_t size, ofstream& of) 
{
    for( size_t i = 0; i < size; ++i) {
        of<<p[i]<<" ";
    }
}

void conjgrad(const Vec<float>& H, float* Hp, const LaplaMat* LM, float* Lp, const Vec<float>& estimate, Vec<float>& x, size_t w, size_t h, float lambda )
{
    cout << "in conjgrad\n";
    size_t size = estimate.getSize();
    Vec<float> r( estimate ), p( estimate ), Ap( size );
    float rsold = Vec<float>::dot( r, r ), alpha = 0.0, rsnew = 0.0;
    //HFilter( Hp, p.getPtr(), H, size, Hp_outfile );
    //LM->run(Lp, p.getPtr(), lambda, tmp_outfile, Lp_outfile);
    //printEstimate( r, size, outfile);
    //getAp( Ap.getPtr(), Hp, Lp, size, w, Ap_outfile );
    //outfile.close();

    for( size_t i = 0; i < 1000; ++i ){
        // Ap = A * p
        cout << i << ' ' << rsold << endl;
        HFilter( Hp, p.getPtr(), H, size);
        // lambda_LFilter( L, image, p.getPtr(), h, w, lambda, radius );
        LM->run(Lp, p.getPtr(), lambda);
        getAp( Ap.getPtr(), Hp, Lp, size);
        alpha = rsold / Vec<float>::dot( p, Ap );
        Vec<float>::add( x, x, p, 1, alpha );
        Vec<float>::add( r, r, Ap, 1, -alpha );
        rsnew = Vec<float>::dot( r, r );
        if( rsnew < 1e-10 ) break;
        // printP(p, size, p_outfile);
        Vec<float>::add( p, r, p, 1, rsnew/rsold );
        rsold = rsnew;
    }
    
}

void vecFloat2uchar( const Vec<float>& F, Vec<uchar>& U )
{
    size_t size = F.getSize();
    float f;
    for( size_t i = 0; i < size; ++i ){
        //f = 255*F[i];
        f = F[i];
        if(f>255) U[i] = (uchar)255;
        else if(f<0) U[i] = (uchar)0;
        else U[i] = uchar(f);
    }
}

void HFilter(float* Hp, const float* p, const Vec<float>& H, const size_t numPixel) {
    for(size_t i = 0; i < numPixel; ++i) {
        if(H[i]) Hp[i] = p[i];
        else Hp[i] = 0;
    //  of<<Hp[i]<<" ";
    }
}

void lambda_LFilter(float* Lp, const float* I_ori, const float* p, const int height, const int width, const float lambda, const int r) {
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

    delete [] tmpI;
}

void propagate2( float* image, float* estimatedBlur, size_t w, size_t h, float lambda, size_t radius, Vec<float>& result )
{
    size_t size = w * h;
    Vec<float> H( size ), ones( size );
    guided_filter gf( image, w, h, radius, 0.00001 );
    constructHE( estimatedBlur, H, result );

    for ( size_t i = 0; i < size; ++i )
    {
        ones[i] = 1;
    }

    cout << size << endl;
    checkHE( H, result );
    cout << 0 << " : " << Vec<float>::dot( ones, H ) << ", " << Vec<float>::dot( ones, result ) << endl;    
    for( size_t i = 0; i < 500; ++i ){
        gf.run(H.getPtr(), H.getPtr());
        gf.run(result.getPtr(), result.getPtr());
        checkHE( H, result );
        Vec<float>::divide( result, result, H );
        checkHE( H, result ); 
        constructHE( result, H, result );
        cout << i << " : " << Vec<float>::dot( ones, H ) << ", " << Vec<float>::dot( ones, result ) << endl;
    }

    // vecFloat2uchar( estimate, result );
}

/*void vecFloat2uchar( const Vec<uchar>& U, Vec<float>& F )
{
    size_t size = U.getSize();
    for( size_t i = 0; i < size; ++i ){
        F[i] = float( U[i] );
    }
}*/

void constructHE( const float* input, Vec<float>& H, Vec<float>& E )
{
    size_t size = H.getSize();
    for( size_t i = 0; i < size; ++i ){
        if( input[i] ){
            H[i] = 1;
            E[i] = input[i];
        }
    }
}

void constructHE( const Vec<float>& input, Vec<float>& H, Vec<float>& E )
{
    size_t size = H.getSize();
    for( size_t i = 0; i < size; ++i ){
        if( input[i] > 1 ) E[i] = 1;
        else if( input[i] < 0 ) E[i] = 0;
        else E[i] = input[i];
        if( E[i] ){
            H[i] = 1;
        }
        else{
            H[i] = 0;
        }
    }

    /*
    for( size_t i = 0; i < size; ++i ){
        E[i] = input[i];
        if( E[i] ){
            H[i] = 1;
        }
        else{
            H[i] = 0;
        }
    }*/
}

//Ap = (H + lamda_L)p = Hp + lamda_L*p
void getAp(float* Ap, const float* Hp, const float* Lp, const int numPixel) {
    for(int i = 0; i < numPixel; ++i) {
        Ap[i] = (float)Hp[i] + Lp[i]; 
    //  of<<Ap[i]<<" ";
    //  if((i+1)%w == 0) of<<endl;
    }
}

void checkHE( const Vec<float>& H, const Vec<float>& E )
{
    size_t size = H.getSize(), HNcount = 0, HBcount = 0, ENcount = 0, EBcount = 0;
    for( size_t i = 0; i < size; ++i ){
        if( H[i] > 1 ) ++HBcount;
        if( H[i] < 0 ) ++HNcount;
        if( E[i] > 255 ) ++EBcount;
        if( E[i] < 0 ) ++ENcount;
    }
    cout << "HB: " << HBcount << " HN: " << HNcount << " EB: " << EBcount << " EN: " << ENcount << endl;

    float maxH, minH, maxE, minE;
    maxH = minH = H[0];
    maxE = minE = E[0];
    for( size_t i = 1; i < size; ++i ){
        if( H[i] > maxH ) maxH = H[i];
        if( H[i] < minH ) minH = H[i];
        if( E[i] > maxE ) maxE = E[i];
        if( E[i] < minE ) minE = E[i];
    }
    cout << "Hmax: " << maxH << " Hmin: " << minH << " Emax: " << maxE << " Emin: " << minE << endl << endl;
}