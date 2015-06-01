#include "fileIO.h"

uchar checkPixelRange(float x)
{
	if(x > 255.f)
		return 255;
	else if(x < 0.f)
		return 0;
	else
		return uchar(x);
}

void saveFloat(float* data, const int width, const int height, string filename) {

	ofstream ofs(filename.c_str(), ios::out);
	ofs << fixed;
	for(int y = 0; y < height; ++y) {
		for(int x = 0; x < width; ++x) {
			ofs << setprecision(5) << data[y*width+x] << " ";
		}
		ofs << endl;
	}
	ofs.close();

}

// Get size of pbm/pgm/ppm format
void sizePGM(int& width, int& height, string filename) {
	
	char ch;
	ifstream ifs(filename.c_str(), ios::binary);
	ch = ifs.get();		// eat "P"
	ch = ifs.get();		// eat "5"
	ifs >> width
		>> height;
	ifs.close();

}

void readPGM(uchar* I, string filename) {
	
	char ch;
	int width, height, tmp;
	ifstream ifs(filename.c_str(), ios::binary);
	if(!ifs) { cerr << "No such file..." << endl; return; }
	ch = ifs.get();		// eat "P"
	ch = ifs.get();		// eat "5"
	ifs >> width
		>> height
		>> tmp;		// tmp will be 255
	if(tmp != 255) {
		cerr << "Input range is wrong!\n";
		return;
	}
	ch = ifs.get();		// eat "\n"
	ifs.read(reinterpret_cast<char*>(I), height*width);
	ifs.close();

}

void readPPM(uchar* I, string filename) {
	
	char ch;
	int width, height, tmp;
	ifstream ifs(filename.c_str(), ios::binary);
	if(!ifs) { cerr << "No such file..." << endl; return; }
	ch = ifs.get();		// eat "P"
	ch = ifs.get();		// eat "6"
	ifs >> width
		>> height
		>> tmp;		// tmp will be 255
	if(tmp != 255) {
		cerr << "Input range is wrong!\n";
		return;
	}
	ch = ifs.get();		// eat "\n"
	ifs.read(reinterpret_cast<char*>(I), height*width*3);
	ifs.close();

}