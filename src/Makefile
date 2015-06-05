all: judy propagate defocus_map

defocus_map: fileIO.o guidedfilter.o propagate.h edge.o defocus.o defocus_map.o
	g++ -Wall -O2 -o defocus_map defocus_map.o fileIO.o guidedfilter.o edge.o defocus.o
defocus_map.o: defocus_map.cpp
	g++ -Wall -c -g -o defocus_map.o defocus_map.cpp
propagate: fileIO.o guidedfilter.o propagate_blur.o propagate.h
	g++ -Wall -O2 -o propagate_blur propagate_blur.o fileIO.o guidedfilter.o

propagate_blur.o:
	g++ -g -c -Wall -o propagate_blur.o propagate_blur.cpp
guidedfilter.o: guidedfilter.cpp guidedfilter.h
	g++ -g -c -Wall -o guidedfilter.o guidedfilter.cpp
judy: judy.o fileIO.o edge.o defocus.o
	g++ -O2 -Wall -o judy judy.o fileIO.o edge.o defocus.o
judy.o: judy.cpp fileIO.h edge.h defocus.h
	g++ -g -c -Wall judy.cpp
fileIO.o: fileIO.cpp fileIO.h
	g++ -g -c -Wall fileIO.cpp
edge.o: edge.cpp edge.h
	g++ -g -c -Wall edge.cpp
defocus.o: defocus.cpp defocus.h
	g++ -g -c -Wall defocus.cpp
test_vec: guidedfilter.o test_vec.cpp
	g++ -O2 -Wall -o test_vec test_vec.cpp guidedfilter.o
clean:
	-rm *.o $(objects) propagate_blur
