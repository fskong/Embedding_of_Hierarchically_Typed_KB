#define ARMA_DONT_USE_CXX11
#include <iostream>
#include <fstream>
#include <armadillo>
#include <time.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace arma;


class Evaluator{
	unordered_map <string, int> entities2index;
	unordered_map <int, string> index2entities;
	unordered_map <string, int> relation2index;
	unordered_map <int ,string> index2relation;
	unordered_map <string, int> type2index;
	unordered_map <string, int> real_type2index;
	unordered_map <int,unordered_set <int>> left_type;
	unordered_map <int,unordered_set <int>> right_type;
	unordered_map <int,unordered_set <int>> entity2real_type;
	unordered_map <int,vector <pair<uvec,uvec>>> relationtype;
	vector<string> testData_instance;
	unordered_set <string> mediator_split;
	vector <int> schema;
	vector<string> real_entity;
	vector<pair<int, uvec> > testData;
	vector<string> dataCvtID;
	unordered_map <int,pair<uvec,uvec>> entitytype;
	vector<vec> A;
	mat BR, NR, ENT;
	mat AE;
	vec DE;
	int num;
	double lambda;
	char detail_out_path[100];
	int ENT_NUM, REL_NUM, DIM;
	int TYPE_NUM;
	int EDGE_NUM = 0;
	void loadFB13K(char *entities_list_path, char *relation_list_path, char *test_data_path, char *split_list, char *real_entity_path, char *detail_path,
					char *type_list_path,char *entity_type_path,char *relation_type_path){
		strcpy(detail_out_path, detail_path);
		FILE *entFile, *relFile, *testFile;
		FILE *typeFile,*enttype,*reltype;
		ifstream f_split(split_list);
		ifstream f_real(real_entity_path);
		string line;
		while(getline(f_real, line)){
			real_entity.push_back(line);
		}
		cout << "real_entity size:\t" << real_entity.size() << endl; 
		f_real.close();
	
		while(getline(f_split, line)){
			mediator_split.insert(line);
		}
		char str[500];
		ENT_NUM = 0, REL_NUM = 0;
		int n;
		entFile = fopen(entities_list_path, "r");
		while (fscanf(entFile, "%s", str) != EOF){
			entities2index[string(str)] = ENT_NUM;
			index2entities[ENT_NUM] = string(str);
			ENT_NUM++;
		}
		fclose(entFile);

		schema.clear();
		relFile = fopen(relation_list_path, "r");
		while (fscanf(relFile, "%s\t%d", str, &n) != EOF){
			schema.push_back(n == 0 ? 2 : n);
			relation2index[string(str)] = REL_NUM;
			index2relation[REL_NUM] = string(str);
			REL_NUM++;
		}
		fclose(relFile);

		testFile = fopen(test_data_path, "r");
		char instance[100];
		while (fscanf(testFile, "%s%s", instance, str) != EOF){
			int index = relation2index[string(str)];
			int cnt = schema[index];
			uvec ent_indices = zeros<uvec>(cnt);
			for (int i = 0; i < cnt; i++){
				fscanf(testFile, "%s", str);
				ent_indices(i) = entities2index[string(str)];
			}
			testData.push_back(pair<int, uvec>(index, ent_indices));
			testData_instance.push_back(string(instance));
		}
		fclose(testFile);
		
		typeFile = fopen(type_list_path, "r");
		while (fscanf(typeFile, "%s", str) != EOF){ 
			int position = string(str).find('.');

			if (type2index.count(string(str).substr(0,position)) == 0)
			{
				type2index[string(string(str).substr(0,position))] = EDGE_NUM;
				EDGE_NUM++;
			}

			type2index[string(str)] = EDGE_NUM;
			EDGE_NUM++;
		}
		type2index["NOTYPE1"] = EDGE_NUM;
		EDGE_NUM++;
		type2index["NOTYPE2"] = EDGE_NUM;
		EDGE_NUM++;
		fclose(typeFile);
		
		typeFile = fopen(type_list_path, "r");
		TYPE_NUM = 0;
		while (fscanf(typeFile, "%s", str) != EOF){  
			real_type2index[string(str)] = TYPE_NUM;
			TYPE_NUM++;
		}
		fclose(typeFile);
		
		enttype = fopen(entity_type_path, "r");
		while (fscanf(enttype, "%s", str) != EOF){
			int index = entities2index[string(str)];
			fscanf(enttype, "%s", str);
			num = atoi(string(str).c_str());
			uvec type_indices1 = zeros<uvec>(num);  
			uvec type_indices2 = zeros<uvec>(num);
			for (int i = 0; i < num ; i++){
				fscanf(enttype, "%s", str);
				type_indices1(i) = type2index[string(string(str).substr(0,string(str).find('.')))];
				type_indices2(i) = type2index[string(str)];
			}
			if (num == 0)
			{
				type_indices1 = zeros<uvec>(1); 
				type_indices2 = zeros<uvec>(1); 
				type_indices1(0) = type2index["NOTYPE1"];
				type_indices2(0) = type2index["NOTYPE2"];

			}
			pair<uvec,uvec> type_indices(type_indices1,type_indices2);

			entitytype[index] = type_indices;
		}
		fclose(enttype);
		
		enttype = fopen(entity_type_path, "r");
		while (fscanf(enttype, "%s", str) != EOF){
			int index = entities2index[string(str)];
			int index_type = 0;
			fscanf(enttype, "%s", str);
			int num = atoi(string(str).c_str());
			for (int i = 0; i < num ; i++){
				fscanf(enttype, "%s", str);
				index_type = real_type2index[str];
				entity2real_type[index].insert(index_type);
			}
		}
		fclose(enttype);
		
		reltype = fopen(relation_type_path, "r");
		int role_num, typ_num;
		while (fscanf(reltype, "%s%d", str, &role_num) != EOF)
		{
			int rel_id = relation2index[string(str)];
			vector <pair<uvec,uvec>> temp_role = vector <pair<uvec,uvec>>(role_num);
			for (int role = 0; role < role_num; role ++) {
				fscanf(reltype, "%d", &typ_num);
				uvec type_indices1 = zeros<uvec>(typ_num);  
				uvec type_indices2 = zeros<uvec>(typ_num);
				for (int i = 0; i < typ_num; i++) {
					fscanf(reltype, "%s", str);
					type_indices1(i) = type2index[string(string(str).substr(0,string(str).find('.')))];
					type_indices2(i) = type2index[string(str)];
				}
				if (typ_num == 0)
				{
					type_indices1 = zeros<uvec>(1); 
					type_indices2 = zeros<uvec>(1); 
					type_indices1(0) = type2index["NOTYPE1"];
					type_indices2(0) = type2index["NOTYPE2"];
				}
				pair<uvec,uvec> type_indices(type_indices1,type_indices2);
				temp_role[role] = type_indices;
			}
			relationtype[rel_id] = temp_role;
		}
		fclose(reltype);
		
		printf("Number of entities: %d, number of relations: %d, number of testing data: %d , number of edge:%d\n", ENT_NUM, REL_NUM, testData.size(),EDGE_NUM);
	}
	void loadMat(char *bias_out, char *entity_out, char *normal_out, char *a_out,char *ae_out,char *de_out){
		BR.load(bias_out);
		ENT.load(entity_out);
		NR.load(normal_out);
		AE.load(ae_out);
		DE.load(de_out);
		DIM = NR.n_rows;
		cout << "DIM:\t" << DIM <<endl;
	        cout << "colum:\t" << NR.n_cols<<endl;	
		FILE *f_transf = fopen(a_out, "rb");
		for (int i = 0; i < REL_NUM; i++){
			vec a = zeros<vec>(schema[i]);
			for (int j = 0; j < schema[i]; j++){
				double tmp;
				fscanf(f_transf, "%lf", &a(j));
			}
			A.push_back(a);
		}	
		
		fclose(f_transf);
	}
	mat project(const mat &_X, const vec &nr){
		return _X - nr * nr.t() * _X;
	}
	double real_lossFn(int rel, const uvec &indices,int role)
	{
		mat X = ENT.cols(indices);
		double temp_loss = 0;

		vec temp = AE.cols(relationtype[rel][role].first).t() * X.col(role) - DE.rows(relationtype[rel][role].first);
		temp_loss = dot(temp,temp);
		temp = AE.cols(relationtype[rel][role].second).t() * X.col(role) - DE.rows(relationtype[rel][role].second);
		temp_loss += dot(temp,temp);

		return temp_loss;
	}
	double lossFn(int rel, const uvec &indices){
		mat X = ENT.cols(indices);
		vec ar = A[rel];
		vec br = BR.col(rel);
		vec tmp = X * ar + br;

		return dot(tmp, tmp);
	}
public:
	Evaluator(char *entities_list_path, char *relation_list_path, char *test_data_path,
		char *entity_out, char *bias_out, char *normal_out, char *a_out, char *split_list, char *real_entity_path, char *detail_path, 
		char *type_list_path,char *entitytype_path,char *ae_out,char *de_out,double _lambda,char *relationtype_path)  {
		loadFB13K(entities_list_path, relation_list_path, test_data_path, split_list, real_entity_path, detail_path,type_list_path,entitytype_path,relationtype_path);
		loadMat(bias_out, entity_out, normal_out, a_out, ae_out,de_out);
		lambda = _lambda;
	}
	
	void evaluate();

};

void Evaluator::evaluate(){
	int rank_head = 1, rank_tail = 1;
	unsigned long long int posTail = 0;
	unsigned long long int posHead = 0;
	int vis_num = 0;
	int rank10numTail = 0;
	int rank10numHead = 0;
	double avgPosTail;
	double avgPosHead;
	double less10perTail;
	double less10perHead;

	for (int k = 0; k < testData.size(); k++){
		rank_head = 1;
		rank_tail = 1;
		int rel = testData[k].first;
		uvec indices = testData[k].second;

		double loss = lossFn(rel, indices);
		double loss0 = real_lossFn(rel, indices,0);
		double loss1 = real_lossFn(rel, indices,1);

		int tail = indices(1);
		int head = indices(0);
		uvec other_tail = indices;
		uvec other_head = indices;
		for (int i = 0; i < ENT_NUM; i++){

			double type_loss = 0;
			double base_loss = 0;
			if (i != tail){
				other_tail(1) = i;
				type_loss = real_lossFn(rel, other_tail,1);
				base_loss = lossFn(rel, other_tail);
				if ( type_loss + base_loss <= loss1 + loss)
				{
					rank_tail++;
				}
			}
			if (i != head){
				other_head(0) = i;
				type_loss = real_lossFn(rel, other_head,0);
				base_loss = lossFn(rel, other_head);
				if (type_loss + base_loss <= loss0 + loss)
				{
					rank_head++;
				}
			}
			
		}
		if (rank_tail <= 10) rank10numTail++;
		if (rank_head <= 10) rank10numHead++;
		posTail += rank_tail;
		posHead += rank_head;
		vis_num += 1;
		less10perTail = rank10numTail * 100.0 / vis_num;
		less10perHead = rank10numHead * 100.0 / vis_num;
		avgPosTail = posTail * 1.0 / vis_num;
		avgPosHead = posHead * 1.0 / vis_num;
		cout << vis_num<<'\t'<<((rank_tail + rank_head)/2.0) << endl;
	}
	printf("\ncase number:%d, hit@10 %f% rank:%f %c", vis_num, (less10perTail + less10perHead) / 2, (avgPosTail + avgPosHead) / 2, 13);
	printf("\n");
}

int main(int argc, char** argv){

	// load data
	char *entities_list_path, *relation_list_path, *test_data_path;
	char *vec_entity, *vec_bias, *vec_normal, *vec_tranf;
	char *type_list_path,*entitytype_path,*relationtype_path;
	char *ae_out,*de_out;
	FILE *entFile, *relFile, *testFile;
	FILE *typeFile, *entyFile;
	Evaluator eva = Evaluator(argv[1], argv[2], argv[11], argv[5], argv[6], argv[7], argv[8], argv[12], argv[13], argv[14], argv[3], argv[4], argv[9], argv[10],atof(argv[15]),argv[16]);
	eva.evaluate();

}
