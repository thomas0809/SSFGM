#include "DataSet.h"
#include "Constant.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MAX_BUF_SIZE 65536

void DataSet::LoadData(const char* data_file, Config* conf)
{
	char  buf[MAX_BUF_SIZE];
	char* eof;

	vector<string>  tokens;
	
	FILE *fin = fopen(data_file, "r");

	for (;;)
	{
		eof = fgets(buf, MAX_BUF_SIZE, fin);
		if (eof == NULL) break;

		// Parse detail information
		tokens = CommonUtil::StringTokenize(buf);

		if (tokens[0] == "#edge") //edge
		{
			int u = atoi(tokens[1].c_str());
			int v = atoi(tokens[2].c_str());

			if (u == v)
				continue;

			DataEdge* curt_edge = new DataEdge();

			curt_edge->a = u;
			curt_edge->b = v;

			curt_edge->edge_type = 0;
			if (tokens.size() >= 4)
				curt_edge->edge_type = edge_type_dict.GetId(tokens[3]);

			edge.push_back(curt_edge);
		}
		else //node
		{
			DataNode* curt_node = new DataNode();

			char label_type = tokens[0][0];
			string label_name = tokens[0].substr(1);

			curt_node->label = label_dict.GetId(label_name);
			if (label_type == '+')
				curt_node->label_type = TRAIN;
			else if (label_type == '*')
				curt_node->label_type = VALID;
			else if (label_type == '?')
				curt_node->label_type = TEST;
			else 
			{
				fprintf(stderr, "Data format wrong! Label must start with +/?/*\n");
				return;
			}
			
			for (int i = 1; i < tokens.size(); i++)
			{
				if (tokens[i][0] == '#')
					break;
				if (conf->has_attrib_value)
				{
					vector<string> key_val = CommonUtil::StringSplit(tokens[i], ':');
					curt_node->attrib.push_back( attrib_dict.GetId(key_val[0]) );
					curt_node->value.push_back( atof(key_val[1].c_str()) );
				}
				else
				{
					curt_node->attrib.push_back( attrib_dict.GetId(tokens[i]) );
					curt_node->value.push_back(1.0);
				}
			}

			curt_node->num_attrib = curt_node->attrib.size();
			node.push_back(curt_node);
		}
	}
	
	num_label = label_dict.GetSize();
	num_attrib_type = attrib_dict.GetSize();
	num_edge_type = edge_type_dict.GetSize();

	num_node = node.size();
	num_edge = edge.size();

	graph = new Graph(num_node);
	for (int i = 0; i < edge.size(); i++)
		graph->add_edge(edge[i]->a, edge[i]->b, edge[i]->edge_type);

	fclose(fin);
	printf("Load Data: %s\n", data_file);
}

void DataSet::LoadDataWithDict(const char* data_file, Config* conf, const MappingDict& ref_label_dict, const MappingDict& ref_attrib_dict, const MappingDict& ref_edge_type_dict)
{
	char  buf[MAX_BUF_SIZE]; 
	char* eof;

	vector<string>  tokens;

	FILE  *fin = fopen(data_file, "r");

	for (;;)
	{
		eof = fgets(buf, MAX_BUF_SIZE, fin);
		if (eof == NULL) break;

		// Parse detail information
		tokens = CommonUtil::StringTokenize(buf);

		if (tokens[0] == "#edge") //edge
		{
			DataEdge* curt_edge = new DataEdge();

			curt_edge->a = atoi(tokens[1].c_str());
			curt_edge->b = atoi(tokens[2].c_str());

			curt_edge->edge_type = 0;
			if (tokens.size() >= 4)
				curt_edge->edge_type = ref_edge_type_dict.GetIdConst( tokens[3] );
			
			if (curt_edge->edge_type >= 0)
				edge.push_back( curt_edge );
		}
		else //node
		{
			DataNode* curt_node = new DataNode();

			char label_type = tokens[0][0];
			string label_name = tokens[0].substr(1);

			curt_node->label = label_dict.GetId(label_name);
			if (label_type == '+')
				curt_node->label_type = TRAIN;
			else if (label_type == '*')
				curt_node->label_type = VALID;
			else if (label_type == '?')
				curt_node->label_type = TEST;
			else 
			{
				fprintf(stderr, "Data format wrong! Label must start with +/?/*\n");
				return;
			}
			
			for (int i = 1; i < tokens.size(); i++)
			{
				if (conf->has_attrib_value)
				{
					vector<string> key_val = CommonUtil::StringSplit(tokens[i], ':');
					int attrib_id = ref_attrib_dict.GetIdConst(key_val[0]);
					if (attrib_id >= 0)
					{
						curt_node->attrib.push_back(attrib_id);
						curt_node->value.push_back(atof(key_val[1].c_str()));
					}
				}
				else
				{
					int attrib_id = ref_attrib_dict.GetIdConst(tokens[i]);
					if (attrib_id >= 0)
					{
						curt_node->attrib.push_back(attrib_id);
						curt_node->value.push_back(1.0);
					}
				}
			}

			curt_node->num_attrib = curt_node->attrib.size();
			node.push_back(curt_node);
		}
	}
	
	num_label = ref_label_dict.GetSize();
	num_attrib_type = ref_attrib_dict.GetSize();
	num_edge_type = ref_edge_type_dict.GetSize();
	
	//if (num_edge_type == 0) num_edge_type = 1;

	fclose(fin);
}
