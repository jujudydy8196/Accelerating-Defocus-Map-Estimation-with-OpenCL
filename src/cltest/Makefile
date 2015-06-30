CXX=g++
EXES=test

CFLAGS=-Wall -c -O2 -std=c++11 
LFLAGS=-lOpenCL -lglog -lgflags

all: $(EXES)

test: main.o cl_helper.o boxfilter.o fileIO.o
	$(CXX) $^ $(LFLAGS) -o $@

%.o: %.cpp %.h
	$(CXX) $(CFLAGS) $< -o $@
 
main.o: main.cpp fileIO.h fileIO.h boxfilter.h cl_helper.h
	$(CXX) $(CFLAGS) $< -o $@

clean:
	-rm *.o $(EXES)
