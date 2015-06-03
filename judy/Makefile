judy: main.o fileIO.o edge.o defocus.o
	g++ -o judy main.o fileIO.o edge.o defocus.o
main.o: main.cpp fileIO.h edge.h defocus.h
	g++ -c main.cpp
fileIO.o: fileIO.cpp fileIO.h
	g++ -c fileIO.cpp
edge.o: edge.cpp edge.h
	g++ -c edge.cpp
defocus.o: defocus.cpp defocus.h fileIO.h
	g++ -c defocus.cpp
clean:
	rm *.o