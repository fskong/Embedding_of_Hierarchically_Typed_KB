objects=m-TransH-Type-train.cpp
 
m-TransH-Type-train m-TransH-Type-test : $(objects) m-TransH-Type-test.cpp
	g++ -std=c++11 -o m-TransH-Type-train $(objects) -O2 -larmadillo
	g++ -std=c++11 -o m-TransH-Type-test  m-TransH-Type-test.cpp -O2 -larmadillo

clean :
	rm m-TransH-Type-train
	rm m-TransH-Type-test
