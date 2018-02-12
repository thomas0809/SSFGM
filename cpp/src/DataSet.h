#pragma once

#include "Util.h"
#include "Config.h"
#include "Graph.h"

#include <string>
#include <vector>
#include <map>
using std::string;
using std::vector;
using std::map;

class DataNode
{
public:
	int                 label_type;
	int                 label;
	int                 num_attrib;
	vector<int>         attrib;
	vector<double>      value;
};

class DataEdge
{
public:
	int                 a, b, edge_type;
};


class DataSet
{
public:

	int num_node;
	int num_edge;
	int num_label;
	int num_attrib_type;
	int num_edge_type;

	vector<DataNode*>   node;
	vector<DataEdge*>   edge;

	Graph* graph;

	MappingDict         label_dict;
	MappingDict         attrib_dict;
	MappingDict         edge_type_dict;

	void LoadData(const char* data_file, Config* conf);
	void LoadDataWithDict(const char* data_file, Config* conf, const MappingDict& ref_label_dict, const MappingDict& ref_attrib_dict, const MappingDict& ref_edge_type_dict);

	~DataSet()
	{
		for (int i = 0; i < node.size(); i++)
			delete node[i];
		for (int i = 0; i < edge.size(); i++)
			delete edge[i];
		delete graph;
	}
};
