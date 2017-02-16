#define ARMA_DONT_USE_CXX11
#include <iostream>
#include <armadillo>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <time.h>
#include <string>
#include <sstream>
#include <boost/functional/hash.hpp>
#include <boost/random.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random/uniform_int.hpp>
using namespace std;
using namespace arma;

float eta;
float type_margin;
int output_per_epoch;
unordered_map <int, unordered_set<int>> entityedge;

typedef boost::minstd_rand  generator_type;  

size_t uvecHash(const uvec &v){      
	return boost::hash_range(v.begin(), v.end());
}
bool eqOp(const uvec &lhs, const uvec &rhs){
	return all((lhs == rhs) == 1);
}
class MFoldEmbedding{
	int DIM, REL_NUM, ENT_NUM, BATCH_SIZE, TRAIN_NUM;
	int EDGE_NUM;
	double alpha, epsilon, gamma, beta;
	double lambda;
	double delta;
	double rellambda,relmargin,distance,reldelta;
	double pos_tloss, neg_tloss, real_neg_tloss;
	double under_margin_num, pos_typ_num, neg_typ_num;
	vector <int> schema;
	//pair<int, uvec> *trainData;
	vector<pair<int, uvec> > trainData;
	unordered_map <int,pair<uvec,uvec>> entitytype;
	unordered_map <int,vector <pair<uvec,uvec>>> relationtype;

	using uvec_set = unordered_set <uvec, decltype(uvecHash)*, decltype(eqOp)* >; 
	unordered_map<int, uvec_set> positive;
	
	//vec *A;
	vector<vec> A, At;
	mat BR, NR, ENT;
	mat Bt, Nt, Et;
	
	mat AE,Ae;
	vec DE,De;
	
	char *bias_out, *entity_out, *normal_out, *a_out;
	char *ae_out, *de_out;


	mat project(const mat &_X, const vec &nr){
		return _X - nr * nr.t() * _X;
	}
	double scoreFn(const mat &Xr, const vec &ar, const vec &br){
		return norm(Xr * ar + br, 2);
	}

	void save(char * out, mat &M){
		FILE *file;
		file = fopen(out, "w");
		for (int i = 0; i < M.n_cols; i++){
			for (int j = 0; j < M.n_rows; j++)
				fprintf(file, "%f\t", M(j, i));
			fprintf(file, "\n");
		}
		fclose(file);
	}
public:
	MFoldEmbedding(unordered_map <int,vector <pair<uvec,uvec>>> &_relationtype,vector <int> &_schema, vector<pair<int, uvec> > &_trainData,
		int trainNum, int dim, int relNum, int entNum, int batchSize,
		double learning_rate, double _epsilon, double margin_gamma, double _beta,
		char *_bias_out, char *_entity_out, char *_normal_out, char *_a_out, double _lambda, int edgeNUM, unordered_map <int,pair<uvec,uvec>> &_entitytype, 
		char *_ae_out, char *_de_out,double _delta,double _reldelta,double _rellambda,double _relmargin,double _distance)
		:bias_out(_bias_out), entity_out(_entity_out), normal_out(_normal_out), a_out(_a_out),
		alpha(learning_rate), epsilon(epsilon), gamma(margin_gamma), beta(_beta),lambda(_lambda), ae_out(_ae_out), 
		de_out(_de_out),delta(_delta),reldelta(_reldelta),rellambda(_rellambda),relmargin(_relmargin),distance(_distance){
		DIM = dim;
		REL_NUM = relNum;
		ENT_NUM = entNum;
		BATCH_SIZE = batchSize;
		TRAIN_NUM = trainNum;
		EDGE_NUM = edgeNUM;
		schema = _schema;
		trainData = _trainData;
		entitytype = _entitytype;
		relationtype = _relationtype;

		
		/*A = vector <vec>(REL_NUM);
		for (int i = 0; i < REL_NUM; i++) {
			//A[i] = randu<vec>(schema[i]);
			int one_size = schema[i];
			A[i] = (randu<vec>(one_size));
		}*/
		
		BR = normalise(randu<mat>(DIM, REL_NUM));
		
		NR = normalise(randu<mat>(DIM, REL_NUM));
	    ENT = normalise(randu<mat>(DIM, ENT_NUM));
		
		AE = normalise(randu<mat>(DIM, EDGE_NUM));
		DE = normalise(randu<vec>(EDGE_NUM));

		loadMat(bias_out, entity_out, normal_out, a_out, ae_out, de_out);

		Nt = NR;
		Bt = BR;
		Et = ENT;
		Ae = AE;
		De = DE;
		for (int i = 0; i < TRAIN_NUM; i++){
			int rel = trainData[i].first;
			uvec indices = trainData[i].second;
			if (positive.count(rel) == 0){  
				positive[rel] = uvec_set(500, uvecHash, eqOp);
			}
			positive[rel].insert(indices);
		}
	}
	~MFoldEmbedding(){
		//delete[] A;
	}
	void saveEmbeddingArma(char *bias_out, char *entity_out, char *normal_out, char *a_out,char *ae_out, char *de_out){
		ENT.save(entity_out);
		BR.save(bias_out);
		NR.save(normal_out);
		AE.save(ae_out);
		DE.save(de_out);
		FILE * file = fopen(a_out, "w");
		for (int i = 0; i < REL_NUM; i++){
			for (int j = 0; j < A[i].n_elem; j++){
				fprintf(file, "%f\t", A[i](j));
			}
			fprintf(file, "\n");
		}
		fclose(file);
	}
	void saveEmbeddingArmaEpoch(char *bias_out, char *entity_out, char *normal_out, char *a_out,char *ae_out, char *de_out, int epoch){
		stringstream ss;
		ss << "." << epoch;
		string epoch_str = ss.str();

		string entity_out_300 = string(entity_out) + epoch_str;
		string bias_out_300 = string(bias_out) + epoch_str;
		string normal_out_300 = string(normal_out) + epoch_str;
		string ae_out_300 = string(ae_out) + epoch_str;
		string de_out_300 = string(de_out) + epoch_str;
		string a_out_300 = string(a_out) + epoch_str;

		ENT.save(entity_out_300);
		BR.save(bias_out_300);
		NR.save(normal_out_300);
		AE.save(ae_out_300);
		DE.save(de_out_300);
		FILE * file = fopen(a_out_300.c_str(), "w");
		for (int i = 0; i < REL_NUM; i++){
			for (int j = 0; j < A[i].n_elem; j++){
				fprintf(file, "%f\t", A[i](j));
			}
			fprintf(file, "\n");
		}
		fclose(file);
	}
	void loadMat(char *bias_out, char *entity_out, char *normal_out, char *a_out,char *ae_out,char *de_out){
		BR.load("result/bias2vec.pretrain");
		ENT.load("result/entity2vec.pretrain");
		NR.load("result/normal2vec.pretrain");
		AE.load("result/ae2vec.pretrain");
		DE.load("result/de2vec.pretrain");
		DIM = NR.n_rows;
		
		cout << "DIM:\t" << DIM <<endl;
	        cout << "colum:\t" << NR.n_cols<<endl;	

		FILE *f_transf = fopen("result/tranf2vec.pretrain", "rb");
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
	double updateGradient(int rel, uvec &posIndices, uvec &negIndices){
		mat Xp = ENT.cols(posIndices);
		mat Xn = ENT.cols(negIndices);
	
		vec ar = A[rel];
		vec br = BR.col(rel);
		vec nr = NR.col(rel);
		mat Xp_r = project(Xp, nr);
		mat Xn_r = project(Xn, nr);
		vec posTmp = Xp_r * ar + br;
		double posLoss = dot(posTmp, posTmp);
		vec negTmp = Xn_r * ar + br;
		double negLoss = dot(negTmp, negTmp);
		if (negLoss - posLoss >= gamma) return 0;

		// compute gradient of relation plane bias vector
		Bt.col(rel) = normalise(Bt.col(rel) -  2 * alpha * (posTmp - negTmp));
		
		// compute gradient of relation plane normal vector
		vec gn = (-2 * dot(posTmp, nr) * Xp * ar - 2 * posTmp * nr.t() * Xp * ar);
		gn -= (-2 * dot(negTmp, nr) * Xn * ar - 2 * negTmp * nr.t() * Xn * ar);
		Nt.col(rel) = normalise(Nt.col(rel) - alpha * gn);

		// compute gradient of entity vectors
		int len = posIndices.n_rows;
		for (int l = 0; l < len; l++){
			vec gp = 2 * ar(l) * (eye<mat>(DIM, DIM) - nr * nr.t()) * posTmp;
			Et.col(posIndices[l]) -= alpha * gp;
			vec gn = 2 * ar(l) * (eye<mat>(DIM, DIM) - nr * nr.t()) * negTmp;
			Et.col(negIndices[l]) += alpha * gn;
		}
		for (int l = 0; l < len; l++){
			Et.col(posIndices[l]) = normalise(Et.col(posIndices[l]));
			Et.col(negIndices[l]) = normalise(Et.col(negIndices[l]));
		}
		return - negLoss +  posLoss + gamma;
	}
	double updateTypeGradient(uvec &posIndices, boost::minstd_rand rng)
	{
		boost::random::uniform_int_distribution<> pick_edge(0, EDGE_NUM - 1);
		mat Xp = ENT.cols(posIndices);
		pos_tloss = 0;
		neg_tloss = 0;
		real_neg_tloss = 0;
		for (int i =0; i < Xp.n_cols; i++)
		{
			pair<uvec,uvec> en;
			en = entitytype[posIndices.row(i)[0]];
			
			
			mat tae = AE.cols(en.first);
			vec tde = DE.rows(en.first);
			vec temp = tae.t() * Xp.col(i) - tde;
			pos_tloss += lambda * dot(temp,temp);
			
			//De.rows(en.first) -= -2 * lambda * delta * temp;
			De.rows(en.first) = normalise(De.rows(en.first) + 2 * lambda * delta * temp);
			
			mat gaet = Xp.col(i) * temp.t();
			Ae.cols(en.first) = normalise(Ae.cols(en.first) - 2 * lambda * delta *  gaet);
			
			vec get = AE.cols(en.first) * temp;
			Et.col(posIndices.row(i)[0]) = normalise(Et.col(posIndices.row(i)[0]) - 2 * lambda * delta *  get);
			
			
			tae = AE.cols(en.second);
			tde = DE.rows(en.second);
			temp = tae.t() * Xp.col(i) - tde;
			pos_tloss += lambda * dot(temp,temp);
			
			//De.rows(en.second) -= -2 * lambda* delta *  temp;
			De.rows(en.second) = normalise(De.rows(en.second) + 2 * lambda* delta *  temp);
			
			gaet = Xp.col(i) * temp.t();
			Ae.cols(en.second) = normalise(Ae.cols(en.second) - 2 * lambda * delta * gaet);
			
			get = AE.cols(en.second) * temp;
			Et.col(posIndices.row(i)[0]) = normalise(Et.col(posIndices.row(i)[0]) - 2 * lambda * delta * get);
			
			pos_typ_num += 2 * tde.n_rows;

			unordered_set<int> edge_set = entityedge[posIndices.row(i)[0]];
			uvec neg_edge(3);   
			for (int i=0;i<3;i++)
			{
				int temp_neg_edge = pick_edge(rng);
				while (edge_set.count(temp_neg_edge) > 0)         
					temp_neg_edge = pick_edge(rng);
				neg_edge(i) = temp_neg_edge;
			}
			tae = AE.cols(neg_edge);
			tde = DE.rows(neg_edge);
			temp = tae.t() * Xp.col(i) - tde;
			double neg_loss = dot(temp,temp);
			neg_typ_num += 3;
			real_neg_tloss += lambda * neg_loss;
			if(neg_loss<type_margin)
			{
				under_margin_num += 3;
				neg_tloss = neg_tloss -lambda*neg_loss + type_margin; 
				De.rows(neg_edge) = normalise(De.rows(neg_edge)-2 * lambda* delta *  temp);
			
				gaet = Xp.col(i) * temp.t();
				Ae.cols(neg_edge) = normalise(Ae.cols(neg_edge) + 2 * lambda * delta * gaet);
			
				get = AE.cols(neg_edge) * temp;
				Et.col(posIndices.row(i)[0]) = normalise(Et.col(posIndices.row(i)[0]) + 2 * lambda * delta * get);
			}

		}
		return pos_tloss + neg_tloss;
	}
	vec createtp(vec a1_prim, vec a2_prim,double d1_prim,double d2_prim)
	{
		vec y = normalise(randu<vec>(DIM));
		double alpha1 = dot(y,a1_prim);
		double alpha2 = dot(y,a2_prim);
		vec u = (alpha1 - d1_prim) * a1_prim + (alpha2 - d2_prim) * a2_prim;
		vec tp = y - u + distance/sqrt(dot(u,u)) * u;
		return tp;
	}
	double updateRelationTypeGradient(int rel, uvec &posIndices, boost::minstd_rand rng)
	{
		int role_num = relationtype[rel].size();
		boost::random::uniform_int_distribution<> pick_role(0, role_num - 1);
				
		int whichrole = pick_role(rng);
		uvec first_layer = relationtype[rel][whichrole].first;
		uvec second_layer = relationtype[rel][whichrole].second;
		
		boost::random::uniform_int_distribution<> pick_type(0, first_layer.n_rows - 1);
		int whichtype = pick_type(rng);
		vec a1 = AE.col(first_layer(whichtype));
		vec a2 = AE.col(second_layer(whichtype));
		double d1 = DE[first_layer(whichtype)];
		double d2 = DE[second_layer(whichtype)];
		
		vec a1_prim = a1;
		vec a2_prim = (- (a1.t() * a2 * a1.t())).t() + a2;
		double d1_prim = d1;
		double d2_prim = - (a1.t() * a2 * d1)[0] + d2;
		
		double dis = 0;
		vec tp;
		int stopflag = 0;
		while(dis < distance)
		{
			tp = createtp(a1_prim,a2_prim,d1_prim,d2_prim);
			dis = 100;
			if (first_layer.n_rows > 3)
				break;
			for (int i = 0 ; i < first_layer.n_rows; i++)
			{
				if (i == whichtype)
					continue;
				vec temp_a1 = AE.col(first_layer(i));
				vec temp_a2 = AE.col(second_layer(i));
				double temp_d1 = DE[first_layer(i)];
				double temp_d2 = DE[second_layer(i)];

				vec temp_a1_prim = temp_a1;
				vec temp_a2_prim = (- (temp_a1.t() * temp_a2 * temp_a1.t())).t() + temp_a2;
				double temp_d1_prim = temp_d1;
				double temp_d2_prim = - (temp_a1.t() * temp_a2 * temp_d1)[0] + temp_d2;
				dis = sqrt(pow((dot(tp,temp_a1_prim)-temp_d1_prim),2) + pow((dot(tp,temp_a2_prim)-temp_d2_prim),2));
				if (dis < distance)
					break;
			}
			stopflag ++;
			if (stopflag ==5)
				break;
		}
		
		mat Xp = ENT.cols(posIndices);
		mat Xn = ENT.cols(posIndices);
		Xn.col(whichrole) = tp;
		vec ar = A[rel];
		vec br = BR.col(rel);
		vec nr = NR.col(rel);
		mat Xp_r = project(Xp, nr);
		mat Xn_r = project(Xn, nr);
		vec posTmp = Xp_r * ar + br;
		double posLoss = dot(posTmp, posTmp);
		vec negTmp = Xn_r * ar + br;
		double negLoss = dot(negTmp, negTmp);
		if (negLoss - posLoss >= relmargin) return 0;

		// compute gradient of relation plane bias vector
		Bt.col(rel) = normalise(Bt.col(rel) -  2 * reldelta * (posTmp - negTmp));
		
		// compute gradient of relation plane normal vector
		vec gn = (-2 * dot(posTmp, nr) * Xp * ar - 2 * posTmp * nr.t() * Xp * ar);
		gn -= (-2 * dot(negTmp, nr) * Xn * ar - 2 * negTmp * nr.t() * Xn * ar);
		Nt.col(rel) = normalise(Nt.col(rel) - reldelta * gn);
		
		return - negLoss +  posLoss + relmargin;
	}
	double orthConstraint(int rel_index){
		double penalty = 0.0;
		vec nr = NR.col(rel_index);
		vec br = BR.col(rel_index);
		double d = dot(nr, br);
		if (d * d > epsilon) {
			Bt.col(rel_index) = normalise(Bt.col(rel_index) - 2 * alpha * beta * d * nr);
			Nt.col(rel_index) = normalise(Nt.col(rel_index) - 2 * alpha * beta * d * br);
			penalty += beta * (d * d - epsilon);
		}
		return penalty;
	}
	
	void updateEmbedding(){
		ENT = Et;
		NR = Nt;
		BR = Bt;
		AE = Ae;
		DE = De;
	}
	uvec negativeSampling(int rel, uvec pos, boost::minstd_rand rng){
		uvec neg = pos;
		boost::random::uniform_int_distribution<> pick_entity(0, ENT_NUM - 1); 

		neg(rand() % neg.n_rows) = pick_entity(rng); 
		while (positive[rel].count(neg) > 0) 
			neg(rand() % neg.n_rows) = pick_entity(rng);
		return neg;
	}
	void train(int num_epoch){
		boost::uniform_int<> uni_dist(0, TRAIN_NUM - 1); 
		generator_type generator(time(0)); 
		boost::variate_generator<generator_type&, boost::uniform_int<> > pick_data(generator, uni_dist);
                                                                                  
		double loss = 0;
		double lossmtransh = 0;
		double losstype = 0;
		double pos_losstype = 0;
		double neg_losstype = 0;
		double real_neg_losstype = 0;
		double temp1 = 0;
		double temp2 = 0;
		double avg_pos_tloss = 0;
		double avg_neg_tloss = 0;
		pos_typ_num = 0;
		neg_typ_num = 0;
		under_margin_num = 0;
		double temp3 = 0;
		double lossrelationtype = 0;

		int num_batch = TRAIN_NUM / BATCH_SIZE + (TRAIN_NUM % BATCH_SIZE ? 1 : 0);
		for (int epoch_index = 1; epoch_index <= num_epoch; epoch_index++){
			printf("%d\tBase:%f\tTyp:%f=%f+%f\trelationtype:%f\tAll:%f\tTrained-neg-typ-edge:%f\tP/N-LosPerEdge:%f/%f\n", 
				epoch_index, lossmtransh, losstype, pos_losstype, neg_losstype,  lossrelationtype,loss, under_margin_num, 
				avg_pos_tloss, avg_neg_tloss);
			if(epoch_index % output_per_epoch == 0)
				saveEmbeddingArmaEpoch(bias_out, entity_out, normal_out, a_out, ae_out, de_out, epoch_index);

			loss = 0;
			lossmtransh = 0;
			losstype = 0;
			pos_losstype = 0;
			neg_losstype = 0;
			real_neg_losstype = 0;
			temp1 = 0;
			temp2 = 0;
			avg_pos_tloss = 0;
			avg_neg_tloss = 0;
			pos_typ_num = 0;
			neg_typ_num = 0;
			under_margin_num = 0;
			temp3 = 0;
			lossrelationtype = 0;
			
			for (int batch_index = 0; batch_index < num_batch; batch_index++){
				for (int i = 0; i < BATCH_SIZE; i++){
					int k = pick_data();
					int rel = trainData[k].first;
					uvec posIndices = trainData[k].second;
					uvec negIndices = negativeSampling(rel, posIndices, generator);
					
					temp1 = updateGradient(rel, posIndices, negIndices);
					lossmtransh += temp1 ;
					loss += temp1;
					temp2 = updateTypeGradient(posIndices, generator);
					losstype += temp2 ;
					pos_losstype += pos_tloss;
					neg_losstype += neg_tloss;
					real_neg_losstype += real_neg_tloss;
					loss += temp2;
					loss += orthConstraint(rel);
					temp3 = updateRelationTypeGradient(rel, posIndices, generator);
					lossrelationtype += temp3;
					loss +=temp3;
				}

				updateEmbedding();
			}
			avg_pos_tloss = pos_losstype / pos_typ_num;
			avg_neg_tloss = real_neg_losstype / neg_typ_num;
		}
		saveEmbeddingArma(bias_out, entity_out, normal_out, a_out, ae_out, de_out);
	}
};
class DataMgr{

public:
	unordered_map <string, int> entities2index;
	unordered_map <string, int> relation2index;
	unordered_map <string, int> type2index;
	vector<int> schema;
	vector<pair<int, uvec>> trainData;

	unordered_map <int,pair<uvec,uvec>> entitytype;
	unordered_map <int,vector <pair<uvec,uvec>>> relationtype;
	int ENT_NUM, REL_NUM;
	int EDGE_NUM;
	
	DataMgr(char *entities_list_path, char *relation_list_path, char *training_data_path, char *type_list_path, char *entity_type_path, char *relation_type_path){
		FILE *entFile, *relFile, *trainFile;
		FILE *typeFile,*enttype,*reltype;
		char str[500];
		int num;
		ENT_NUM = 0, REL_NUM = 0;
		EDGE_NUM = 0;
		int n;
		entFile = fopen(entities_list_path, "r");
		while (fscanf(entFile, "%s", str) != EOF){
			entities2index[string(str)] = ENT_NUM;
			ENT_NUM++;
		}
		fclose(entFile);
		
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

		schema.clear();
		relFile = fopen(relation_list_path, "r");
		while (fscanf(relFile, "%s\t%d", str, &n) != EOF){ 
			schema.push_back(n == 0 ? 2 : n);
			relation2index[string(str)] = REL_NUM;
			REL_NUM++;
		}
		fclose(relFile);

		trainFile = fopen(training_data_path, "r");
		while (fscanf(trainFile, "%s", str) != EOF){
			int index = relation2index[string(str)];
			int cnt = schema[index];
			uvec ent_indices = zeros<uvec>(cnt);
			for (int i = 0; i < cnt; i++){
				fscanf(trainFile, "%s", str);
				ent_indices(i) = entities2index[string(str)];
			}
			trainData.push_back(pair<int, uvec>(index, ent_indices));
		}
		fclose(trainFile);
		
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

			//construct the structure for negative type instance
			uvec edge = join_cols(type_indices1, type_indices2);
			unordered_set<int> edge_set;
			for (int i = 0;i < edge.n_rows;i++)
			{
				edge_set.insert(edge(i));
			}
			entityedge[index] = edge_set;
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
		
		printf("Number of entities: %d, number of relations: %d, number of training data: %d, number of edge: %d\n", ENT_NUM, REL_NUM, trainData.size(),EDGE_NUM);
	}
};
int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}
int main(int argc, char** argv)
{
	int i, dim, num_epoch, batch_size;
	double learning_rate, margin_gamma, epsilon, beta;
	double lambda;
	double delta;
	double rellambda,relmargin,distance,reldelta;

	char *entities_list_path, *relation_list_path, *training_data_path, schema_path;
	char *type_list_path, *entity_type_path,*relation_type_path;
	
	char *bias_out, *entity_out, *normal_out, *a_out, *split_list;
	char *ae_out, *de_out;
	if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epoch", argc, argv)) > 0) num_epoch = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-batch", argc, argv)) > 0) batch_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-lr", argc, argv)) > 0) learning_rate = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin_gamma = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-epsilon", argc, argv)) > 0) epsilon = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-beta", argc, argv)) > 0) beta = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-ll", argc, argv)) > 0) lambda = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-del", argc, argv)) > 0) delta = atof(argv[i + 1]);

	if ((i = ArgPos((char *)"-entity", argc, argv)) > 0) entities_list_path = argv[i + 1];
	if ((i = ArgPos((char *)"-rel", argc, argv)) > 0) relation_list_path = argv[i + 1];
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) training_data_path = argv[i + 1];
	if ((i = ArgPos((char *)"-type", argc, argv)) > 0) type_list_path = argv[i + 1];
	if ((i = ArgPos((char *)"-entitytype", argc, argv)) > 0) entity_type_path = argv[i + 1];
	if ((i = ArgPos((char *)"-relationtype", argc, argv)) > 0) relation_type_path = argv[i + 1];

	if ((i = ArgPos((char *)"-bias_out", argc, argv)) > 0) bias_out = argv[i + 1];
	if ((i = ArgPos((char *)"-entity_out", argc, argv)) > 0) entity_out = argv[i + 1];
	if ((i = ArgPos((char *)"-normal_out", argc, argv)) > 0) normal_out = argv[i + 1];
	if ((i = ArgPos((char *)"-a_out", argc, argv)) > 0) a_out = argv[i + 1];
	if ((i = ArgPos((char *)"-ae_out", argc, argv)) > 0) ae_out = argv[i + 1];
	if ((i = ArgPos((char *)"-de_out", argc, argv)) > 0) de_out = argv[i + 1];
	if ((i = ArgPos((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-type_margin", argc, argv)) > 0) type_margin = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output_per_epoch", argc, argv)) > 0) output_per_epoch = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-relmargin", argc, argv)) > 0) relmargin = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-distance", argc, argv)) > 0) distance = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-rellambda", argc, argv)) > 0) rellambda = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-reldelta", argc, argv)) > 0) reldelta = atof(argv[i + 1]);
	
	//printf("dim = %d num_epoch = %d batch= %d lr = %f margin = %f epsilon = %f beta = %f\n",
		//dim, num_epoch, batch_size, learning_rate, margin_gamma, epsilon, beta);
	printf("dim = %d num_epoch = %d batch= %d lr = %f margin = %f epsilon = %f beta = %f lambda = %f delta = %f type_margin = %f\n",
		dim, num_epoch, batch_size, learning_rate, margin_gamma, epsilon, beta,lambda,delta, type_margin);	
	printf("relmargin = %f distance = %f rellambda= %f reldelta = %f\n",
		relmargin, distance, rellambda, reldelta);
		
	DataMgr dm = DataMgr(entities_list_path, relation_list_path, training_data_path, type_list_path, entity_type_path,relation_type_path);
	
	MFoldEmbedding model = MFoldEmbedding(dm.relationtype,dm.schema, dm.trainData,
		dm.trainData.size(), dim, dm.REL_NUM, dm.ENT_NUM, batch_size,
		learning_rate, epsilon, margin_gamma, beta,
		bias_out, entity_out, normal_out, a_out, lambda, dm.EDGE_NUM, dm.entitytype,ae_out,de_out,delta,
		reldelta,rellambda,relmargin,distance);
	model.train(num_epoch);

	//model.saveEmbedding(bias_out, entity_out, normal_out, a_out);
	return 0;
}
