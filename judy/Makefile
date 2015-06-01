all: judy propagete

propagete: fileIO.o guidedfilter.o

guidedfilter.o: guidedfilter.cpp guidedfilter.h
	g++ -c -Wall -o guidedfilter.o guidedfilter.cpp
judy: judy.o fileIO.o edge.o defocus.o
	g++ -O2 -Wall -o judy judy.o fileIO.o edge.o defocus.o
judy.o: judy.cpp fileIO.h edge.h defocus.h
	g++ -c -Wall judy.cpp
fileIO.o: fileIO.cpp fileIO.h
	g++ -c -Wall fileIO.cpp
edge.o: edge.cpp edge.h
	g++ -c -Wall edge.cpp
defocus.o: defocus.cpp defocus.h
	g++ -c -Wall defocus.cpp