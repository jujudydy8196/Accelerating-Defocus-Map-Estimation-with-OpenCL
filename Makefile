all:
	g++ -Wall -O2 -o propagate_blur guidedfilter.cpp main.cpp fileIO.cpp