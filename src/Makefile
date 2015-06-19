CXX=g++
EXES=judy propagate defocus_map test_vec cl_test

CFLAGS=-Wall -c -O2 -std=c++11 
LFLAGS=-lOpenCL -lglog -lgflags

all: $(EXES)


judy: judy.o fileIO.o edge.o defocus.o
	$(CXX) $^ $(LFLAGS) -o $@

propagate_blur: fileIO.o guidedfilter.o propagate_blur.o propagate.o
	$(CXX) $? $(LFLAGS) -o $@

defocus_map: fileIO.o guidedfilter.o edge.o defocus.o defocus_map.o propagatecl.o cl_helper.o vec.o propagate.o
	$(CXX) $^ $(LFLAGS) -o $@

test_vec: guidedfilter.o test_vec.o
	$(CXX) $^ $(LFLAGS) -o $@

cl_test: cl_test.o cl_helper.o propagatecl.o propagate.o
	$(CXX) $^ $(LFLAGS) -o $@


%.o: %.cpp %.h
	$(CXX) $(CFLAGS) $< -o $@

defocus_map.o: defocus_map.cpp propagate.h vec.h
	$(CXX) $(CFLAGS) $< -o $@

propagate_blur.o: propagate_blur.cpp propagate.h vec.h
	$(CXX) $(CFLAGS) $< -o $@

judy.o: judy.cpp fileIO.h edge.h defocus.h
	$(CXX) $(CFLAGS) $< -o $@

test_vec.o: test_vec.cpp vec.h
	$(CXX) $(CFLAGS) $< -o $@

cl_test.o: cl_test.cpp
	$(CXX) $(CFLAGS) $< -o $@


clean:
	-rm *.o $(EXES)
