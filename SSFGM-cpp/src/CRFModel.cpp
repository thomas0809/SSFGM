#include "CRFModel.h"
#include "Constant.h"
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MAX_BUF_SIZE 65536

void CRFModel::Init()
{
    data = new DataSet();
    data->LoadData(conf->train_file.c_str(), conf);

    num_node = data->num_node;
    num_edge = data->num_edge;
    num_label = data->num_label;
    num_attrib_type = data->num_attrib_type;
    num_edge_type = data->num_edge_type;

    printf("num_node = %d\n", data->num_node);
    printf("num_edge = %d\n", data->num_edge);
    printf("num_label = %d\n", data->num_label);
    printf("num_edge_type = %d\n", data->num_edge_type);
    printf("num_attrib_type = %d\n", data->num_attrib_type);

    GenParameter(conf->directed);

    if (conf->method == "LBP")
        SetupFactorGraphs();
    if (conf->method == "MH" || conf->method == "TCMH")
        MH_Init();

    probability = new double*[num_node];
    for (int i = 0; i < num_node; i++)
        probability[i] = new double[num_label];

    unlabeled.clear();
    test.clear();
    valid.clear();
    for (int i = 0; i < num_node; i++)
    {
        if (data->node[i]->label_type != KNOWN_LABEL)
        {
            unlabeled.push_back(i);
            if (data->node[i]->label_type == VALID) 
                valid.push_back(i);
            else 
                test.push_back(i);
        }
    }
}

void CRFModel::GenParameter(bool directed)
{
    num_parameter = 0;

    // state feature: f(y, x)
    num_attrib_parameter = num_label * num_attrib_type;
    num_parameter += num_attrib_parameter;

    // edge feature: f(edge_type, y1, y2)
    edge_parameter_offset.clear();
    int offset = 0;
    if (!directed)
    {
        for (int y1 = 0; y1 < num_label; y1++)
            for (int y2 = y1; y2 < num_label; y2++)
            {
                edge_parameter_offset.insert( make_pair(y1 * num_label + y2, offset) );
                edge_parameter_offset.insert( make_pair(y2 * num_label + y1, offset) );
                offset ++;
            }
    }
    else
    {
        for (int y1 = 0; y1 < num_label; y1++)
            for (int y2 = 0; y2 < num_label; y2++)
            {
                edge_parameter_offset.insert( make_pair(y1 * num_label + y2, offset) );
                offset ++;
            }
    }
    num_edge_parameter_each_type = offset;
    num_parameter += num_edge_type * num_edge_parameter_each_type;

    lambda = new double[num_parameter];
    // Initialize parameters
    for (int i = 0; i < num_parameter; i++)
        lambda[i] = 0.0;

    
    printf("num_parameter = %d\n", num_parameter);
}

void CRFModel::Clean()
{
    if (lambda) 
        delete[] lambda;
    if (probability)
    {
        for (int i = 0; i < num_node; i++)
            delete[] probability[i];
        delete[] probability;
    }
    if (factor_graph)
        delete factor_graph;
    for (int i = 0; i < num_edge_type; i++)
        delete func_list[i];
    delete[] func_list;
}

void CRFModel::LoadModel(const char* filename)
{
    FILE* fin = fopen(filename, "r");
    char buf[MAX_BUF_SIZE];
    vector<string> tokens;
    for (;;)
    {
        if (fgets(buf, MAX_BUF_SIZE, fin) == NULL)
            break;
        tokens = CommonUtil::StringTokenize(buf);
        if (tokens[0] == "#node")
        {
            int class_id = data->label_dict.GetIdConst(tokens[1]);
            int feat_id = data->attrib_dict.GetIdConst(tokens[2]);
            int pid = GetAttribParameterId(class_id, feat_id);
            double value = atof(tokens[3].c_str());
            lambda[pid] = value;
        }
        if (tokens[0] == "#edge")
        {
            int type_id = data->edge_type_dict.GetIdConst(tokens[1]);
            int id1 = data->label_dict.GetIdConst(tokens[2]);
            int id2 = data->label_dict.GetIdConst(tokens[3]);
            int pid = GetEdgeParameterId(type_id, id1, id2);
            double value = atof(tokens[4].c_str());
            lambda[pid] = value;
        }
    }
    fclose(fin);
    printf("Load %s finished.\n", filename);
}

void CRFModel::SaveModel(const char* filename)
{
    FILE* fout = fopen(filename, "w");
    for (int i = 0; i < num_label; i++)
    {
        string cl = data->label_dict.GetKeyWithId(i);
        for (int j = 0; j < num_attrib_type; j++)
        {
            string feature = data->attrib_dict.GetKeyWithId(j);
            int pid = GetAttribParameterId(i, j);
            fprintf(fout, "#node %s %s %f\n", cl.c_str(), feature.c_str(), lambda[pid]);
        }
    }
    for (int T = 0; T < num_edge_type; T++)
    {
        string c0 = data->edge_type_dict.GetKeyWithId(T);
        for (int i = 0; i < num_label; i++)
        {
            string c1 = data->label_dict.GetKeyWithId(i);
            for (int j = i; j < num_label; j++)
            {
                string c2 = data->label_dict.GetKeyWithId(j);
                int pid = GetEdgeParameterId(T, i, j);
                fprintf(fout, "#edge %s %s %s %f\n", c0.c_str(), c1.c_str(), c2.c_str(), lambda[pid]);
            }
        }
    }
    fclose(fout);
}

void CRFModel::Estimate()
{
    if (conf->src_model_file != "") 
        LoadModel(conf->src_model_file.c_str());

    printf("Start Training...\n");
    fflush(stdout);

    if (conf->method == "LBP") 
        LBP_Train();
    else if (conf->method == "MH")
    {
        if (conf->state_file != "") 
            LoadPrediction(conf->state_file.c_str());
        MH_Train();
    }
    else if (conf->method == "TCMH")
    {
        if (conf->state_file != "") 
            LoadPrediction(conf->state_file.c_str());
        TCMH_Train();
    }
    else
    {
        printf("Method error!\n");
        return;
    }
    SavePrediction(conf->pred_file.c_str());
    SaveModel(conf->dst_model_file.c_str());
}

void CRFModel::Inference()
{
    printf("num_label = %d\n", num_label);
    printf("num_edge_type = %d\n", num_edge_type);
    printf("num_attrib_type = %d\n", num_attrib_type);
    
    LoadModel(conf->src_model_file.c_str());
    
    if (conf->method == "LBP")
    {
        LBP_Test();
    }
    else if ((conf->method == "MH") || (conf->method == "TCMH"))
    {
        if (conf->state_file != "") 
            LoadPrediction(conf->state_file.c_str());
        MH_Test(conf->max_infer_iter);
    }
    else
    {
        printf("Method error!\n");
        return;
    }
    SavePrediction(conf->pred_file.c_str());
}

void CRFModel::SavePrediction(const char* filename)
{
    FILE* fout = fopen(filename, "w");
    
    for (int i = 0; i < num_label; i++)
        fprintf(fout, "%s ", data->label_dict.GetKeyWithId(i).c_str());
    fprintf(fout, "\n");

    for (int u = 0; u < num_node; u++)
    {
        for (int k = 0; k < num_label; k++)
            fprintf(fout, "%.3f ", probability[u][k]);
        fprintf(fout, "\n");
    }
    fclose(fout);
}

void CRFModel::LoadPrediction(const char* filename)
{
    FILE* fin = fopen(filename, "r");
    char buf[MAX_BUF_SIZE];
    vector<string> tokens;

    char *c = fgets(buf, MAX_BUF_SIZE, fin);
    tokens = CommonUtil::StringTokenize(buf);
    vector<int> labels;
    for (int i = 0; i < tokens.size(); i++)
        labels.push_back(data->label_dict.GetIdConst(tokens[i]));

    state.clear();
    state.assign(num_node, 0);
    for (int i = 0; i < num_node; i++)
    {
        if (fgets(buf, MAX_BUF_SIZE, fin) == NULL)
            break;
        if (data->node[i]->label_type == KNOWN_LABEL)
        {
            state[i] = data->node[i]->label;
            continue;
        }
        tokens = CommonUtil::StringTokenize(buf);
        double max_p = 0;
        int label = 0;
        for (int j = 0; j < tokens.size(); j++)
        {
            double p = atof(tokens[j].c_str());
            if (p > max_p)
            {
                max_p = p;
                label = labels[j];
            }
        }
        state[i] = label;
    }
    fclose(fin);
    printf("Load %s finished.\n", filename);
}













// void CRFModel::SelfEvaluate()
// {
//  int ns = data->num_sample;
//  int tot, hit;

//  tot = hit = 0;
//  for (int s = 0; s < ns; s++)
//  {
//      DataSample* sample = train_data->sample[s];
//      FactorGraph* factor_graph = &sample_factor_graph[s];
        
//      int n = sample->num_node;
//      int m = sample->num_edge;
        
//      factor_graph->InitGraph(n, m, num_label);
//      // Add edge info
//      for (int i = 0; i < m; i++)
//      {
//          factor_graph->AddEdge(sample->edge[i]->a, sample->edge[i]->b, func_list[sample->edge[i]->edge_type]);
//      }        
//      factor_graph->GenPropagateOrder();

//      factor_graph->ClearDataForMaxSum();

//      for (int i = 0; i < n; i++)
//      {
//          double* p_lambda = lambda;

//          for (int y = 0; y < num_label; y++)
//          {
//              double v = 1.0;
//              for (int t = 0; t < sample->node[i]->num_attrib; t++)
//                  v *= exp(p_lambda[sample->node[i]->attrib[t]] * sample->node[i]->value[t]);
//              factor_graph->SetVariableStateFactor(i, y, v);

//              p_lambda += num_attrib_type;
//          }
//      }    

//      factor_graph->MaxSumPropagation(conf->max_infer_iter);

//      int* inf_label = new int[n];
//      for (int i = 0; i < n; i++)
//      {
//          int ybest = -1;
//          double vbest, v;

//          for (int y = 0; y < num_label; y++)
//          {
//              v = factor_graph->var_node[i].state_factor[y];
//              for (int t = 0; t < factor_graph->var_node[i].neighbor.size(); t++)
//                  v *= factor_graph->var_node[i].belief[t][y];
//              if (ybest < 0 || v > vbest)
//                  ybest = y, vbest = v;
//          }

//          inf_label[i] = ybest;
//      }

//      int curt_tot, curt_hit;
//      curt_tot = curt_hit = 0;
//      for (int i = 0; i < n; i++)
//      {   
//          curt_tot ++;
//          if (inf_label[i] == sample->node[i]->label) curt_hit++;
//      }
        
//      printf("Accuracy %4d / %4d : %.6lf\n", curt_hit, curt_tot, (double)curt_hit / curt_tot);
//      hit += curt_hit;
//      tot += curt_tot;

//      delete[] inf_label;
//  }

//  printf("Overall Accuracy %4d / %4d : %.6lf\n", hit, tot, (double)hit / tot);
// }

