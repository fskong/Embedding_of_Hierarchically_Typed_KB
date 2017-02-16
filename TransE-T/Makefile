objects=TransE-Type-train.cpp
 
TransE-Type-train TransE-Type-test : $(objects) TransE-Type-test.cpp
	g++ -std=c++11 -o TransE-Type-train $(objects) -O2 -larmadillo
	g++ -std=c++11 -o TransE-Type-test  TransE-Type-test.cpp -O2 -larmadillo

clean :
	rm TransE-Type-train
	rm TransE-Type-test
