objects=TransH-Type-train.cpp
 
TransH-Type-train TransH-Type-test : $(objects) TransH-Type-test.cpp
	g++ -std=c++11 -o TransH-Type-train $(objects) -O2 -larmadillo
	g++ -std=c++11 -o TransH-Type-test  TransH-Type-test.cpp -O2 -larmadillo

clean :
	rm TransH-Type-train
	rm TransH-Type-test
