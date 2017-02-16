#include <iostream>
#include <cstdio>
#include <iomanip>
using namespace std;

void sumup(string path){
    FILE* fin = fopen(path.c_str(),"r");
    double rank=0, hit=0, size=0;
    double split_rank, split_hit, split_size;
    while (fscanf(fin,"%lf",&split_size)==1)
    {
        fscanf(fin,"%lf %lf", &split_hit, &split_rank);
        size += split_size;
        rank += split_size * split_rank;
        hit += split_size * split_hit;
    }
    rank /= size;
    hit /= size;
	cout << setiosflags(ios::fixed);
    cout << setprecision(6) << "hit: " << hit << endl;
    cout << "rank: " << rank << endl; 
}

int main(int argc,char**argv)
{
    string path = argv[1];
    sumup(path);
}

