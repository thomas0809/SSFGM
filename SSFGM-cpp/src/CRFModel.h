#pragma once

#include "Config.h"
#include "DataSet.h"
#include "FactorGraph.h"



class CRFModel
{
public:
	CRFModel(Config* conf)
	{
		this->conf = conf;
		Init();
	}

	~CRFModel()
	{
		Clean();
	}

	void Estimate();
	void Inference();

private:
	Config*     conf;
	DataSet*    data;

	int         num_node;
	int         num_edge;
	int         num_label;
	int         num_attrib_type;
	int         num_edge_type;
	
	int         num_parameter;
	int         num_attrib_parameter;
	int         num_edge_parameter_each_type;
	map<int, int>   edge_parameter_offset;

	EdgeFactorFunction**   func_list;

	double          *lambda;
	double          **probability;
	FactorGraph     *factor_graph;

	vector<int> state;
	vector<int> unlabeled;
	vector<int> test;
	vector<int> valid;

	void Init();
	void Clean();

	void SaveModel(const char* file_name);
	void LoadModel(const char* file_name);

	void SavePrediction(const char* filename);
	void LoadPrediction(const char* filename);

	// for LBP
	void GenParameter(bool directed);
	void SetupFactorGraphs();

	void LBP_Train();
	void LBP_Test();

	double CalcGradient(double* gradient);


	void MH_Train();
	double MH_Test(int max_iter = 0);
	void MH_Init();
	double CalcLikelihood(int u, vector<int> &_state, map<int,double> *gradient);

	void TCMH_Train();
	bool train1_sample(int type, int center, int ynew, double p, vector<int>& _state, map<int,double>& _gradient);


	int GetAttribParameterId(int y, int x)
	{
		return y * num_attrib_type + x;
	}

	int GetEdgeParameterId(int edge_type, int a, int b)
	{ 
		int offset = edge_parameter_offset[a * num_label + b];
		return num_attrib_parameter + edge_type * num_edge_parameter_each_type + offset;
	}
};

