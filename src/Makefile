CXX=g++
EXES=judy propagate defocus_map test_vec

CFLAGS=-Wall -c -O2 -std=c++11 
LFLAGS=-lOpenCL -lglog -lgflags

all: $(EXES)


judy: judy.o fileIO.o edge.o defocus.o
	$(CXX) $? $(LFLAGS) -o $@

propagate_blur: fileIO.o guidedfilter.o propagate_blur.o
	$(CXX) $? $(LFLAGS) -o $@

defocus_map: fileIO.o guidedfilter.o edge.o defocus.o defocus_map.o
	$(CXX) $? $(LFLAGS) -o $@

test_vec: guidedfilter.o test_vec.o
	$(CXX) $? $(LFLAGS) -o $@


%.o: %.cpp %.h
	$(CXX) $(CFLAGS) $< -o $@

defocus_map.o: defocus_map.cpp propagate.h vec.h
	$(CXX) $(CFLAGS) $< -o $@

propagate_blur.o: propagate_blur.cpp propagate.h vec.h
	$(CXX) $(CFLAGS) $< -o $@

judy.o: judy.cpp fileIO.h edge.h defocus.h
	$(CXX) $(CFLAGS) $< -o $@

test_vec.o: test_vec.cpp vec.h
	g++ -O2 -Wall -o test_vec test_vec.cpp guidedfilter.o


clean:
	-rm *.o $(EXES)
