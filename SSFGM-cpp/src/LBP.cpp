#include "CRFModel.h"
#include "Constant.h"
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void CRFModel::SetupFactorGraphs()
{
    factor_graph = new FactorGraph();
    factor_graph->InitGraph(num_node, num_edge, num_label);

    // Add node info
    for (int i = 0; i < num_node; i++)
    {
        factor_graph->SetVariableLabel(i, data->node[i]->label);
        factor_graph->var_node[i].label_type = data->node[i]->label_type;
    }

    // Add edge info
    double* p_lambda = lambda + num_attrib_parameter;
    func_list = new EdgeFactorFunction*[num_edge_type];
    for (int i = 0; i < num_edge_type; i++)
    {
        func_list[i] = new EdgeFactorFunction(num_label, p_lambda, &edge_parameter_offset);
        p_lambda += num_edge_parameter_each_type;
    }

    for (int i = 0; i < num_edge; i++)
    {
        factor_graph->AddEdge(data->edge[i]->a, data->edge[i]->b, func_list[data->edge[i]->edge_type]);
    }

    factor_graph->GenPropagateOrder();

    printf("Setup Factor Graph.\n");
}

void CRFModel::LBP_Train()
{    
    double* gradient;
    double  f;          // log-likelihood

    gradient = new double[num_parameter + 1];

    // Variable for optimization
    double  eps = conf->eps;
    double  old_f = 0.0;

    // Main-loop of CRF
    // Paramater estimation via Gradient Descend

    double start_time, end_time;

    for (int iter = 0; iter < conf->max_iter; iter++)
    {
        start_time = clock() / (double)CLOCKS_PER_SEC;

        // Step A. Calc gradient and log-likehood of the local datas
        f = CalcGradient(gradient);

        // Step B. Opitmization by Gradient Descend
        printf("[Iter %3d] log-likelihood : %.8lf\n", iter, f);
        fflush(stdout);

        // If diff of log-likelihood is small enough, break.
        if (fabs(old_f - f) < eps) break;
        old_f = f;

        // Normalize Graident
        double g_norm = 0.0;
        for (int i = 0; i < num_parameter; i++)
            g_norm += gradient[i] * gradient[i];
        g_norm = sqrt(g_norm);
            
        if (g_norm > 1e-8)
        {
            for (int i = 0; i < num_parameter; i++)
                gradient[i] /= g_norm;
        }

        for (int i = 0; i < num_parameter; i++)
            lambda[i] += gradient[i] * conf->gradient_step;

        if (iter % conf->eval_interval == 0)
            LBP_Test();

        end_time = clock() / (double)CLOCKS_PER_SEC;

        // printf("!!! Time cost = %.6lf\n", end_time - start_time);
        // fflush(stdout);

    }

    LBP_Test();

    delete[] gradient;
}

double CRFModel::CalcGradient(double* gradient)
{   
    int n = num_node;
    int m = num_edge;
    
    //****************************************************************
    // Belief Propagation 1: labeled data are given.
    //****************************************************************

    factor_graph->labeled_given = true;
    factor_graph->ClearDataForSumProduct();
    
    // Set state_factor
    for (int i = 0; i < n; i++)
    {
        double* p_lambda = lambda;
        for (int y = 0; y < num_label; y++)
        {
            if (data->node[i]->label_type == KNOWN_LABEL && y != data->node[i]->label)
            {
                factor_graph->SetVariableStateFactor(i, y, 0);
            }
            else
            {
                double v = 1;
                for (int t = 0; t < data->node[i]->num_attrib; t++)
                    v *= exp(p_lambda[data->node[i]->attrib[t]] * data->node[i]->value[t]); 
                factor_graph->SetVariableStateFactor(i, y, v);
            }
            p_lambda += num_attrib_type;
        }
    }
    
    factor_graph->BeliefPropagation(conf->max_infer_iter);
    factor_graph->CalculateMarginal();    

    /***
    * Gradient = E_{Y|Y_L} f_i - E_{Y} f_i
    */

    // calc gradient part : + E_{Y|Y_L} f_i
    for (int i = 0; i < n; i++)
    {
        for (int y = 0; y < num_label; y++)
        {
            for (int t = 0; t < data->node[i]->num_attrib; t++)
                gradient[GetAttribParameterId(y, data->node[i]->attrib[t])] += data->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }

    for (int i = 0; i < m; i++)
    {
        for (int a = 0; a < num_label; a++)
            for (int b = 0; b < num_label; b++)
            {
                gradient[GetEdgeParameterId(data->edge[i]->edge_type, a, b)] += factor_graph->factor_node[i].marginal[a][b];
            }
    }

    //****************************************************************
    // Belief Propagation 2: labeled data are not given.
    //****************************************************************

    factor_graph->ClearDataForSumProduct();
    factor_graph->labeled_given = false;

    for (int i = 0; i < n; i++)
    {
        double* p_lambda = lambda;
        for (int y = 0; y < num_label; y++)
        {
            double v = 1;
            for (int t = 0; t < data->node[i]->num_attrib; t++)
                v *= exp(p_lambda[data->node[i]->attrib[t]] * data->node[i]->value[t]);
            factor_graph->SetVariableStateFactor(i, y, v);
            p_lambda += num_attrib_type;
        }
    }    

    factor_graph->BeliefPropagation(conf->max_infer_iter);
    factor_graph->CalculateMarginal();
    
    // calc gradient part : - E_{Y} f_i
    for (int i = 0; i < n; i++)
    {
        for (int y = 0; y < num_label; y++)
        {
            for (int t = 0; t < data->node[i]->num_attrib; t++)
                gradient[GetAttribParameterId(y, data->node[i]->attrib[t])] -= data->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }
    for (int i = 0; i < m; i++)
    {
        for (int a = 0; a < num_label; a++)
            for (int b = 0; b < num_label; b++)
            {
                gradient[GetEdgeParameterId(data->edge[i]->edge_type, a, b)] -= factor_graph->factor_node[i].marginal[a][b];
            }
    }
    
    // Calculate gradient & log-likelihood
    double f = 0.0, Z = 0.0;

    // \sum \lambda_i * f_i
    for (int i = 0; i < n; i++)
    {
        int y = data->node[i]->label;
        for (int t = 0; t < data->node[i]->num_attrib; t++)
            f += lambda[this->GetAttribParameterId(y, data->node[i]->attrib[t])] * data->node[i]->value[t];
    }
    for (int i = 0; i < m; i++)
    {
        int a = data->node[data->edge[i]->a]->label;
        int b = data->node[data->edge[i]->b]->label;        
        f += lambda[this->GetEdgeParameterId(data->edge[i]->edge_type, a, b)];
    }

    // calc log-likelihood
    //  using Bethe Approximation
    for (int i = 0; i < n; i++)
    {
        for (int y = 0; y < num_label; y++)
        {
            for (int t = 0; t < data->node[i]->num_attrib; t++)
                Z += lambda[this->GetAttribParameterId(y, data->node[i]->attrib[t])] * data->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }
    for (int i = 0; i < m; i++)
    {
        for (int a = 0; a < num_label; a++)
            for (int b = 0; b < num_label; b++)
            {
                Z += lambda[this->GetEdgeParameterId(data->edge[i]->edge_type, a, b)] * factor_graph->factor_node[i].marginal[a][b];
            }
    }
    // Edge entropy
    for (int i = 0; i < m; i++)
    {
        double h_e = 0.0;
        for (int a = 0; a < num_label; a++)
            for (int b = 0; b < num_label; b++)
            {
                if (factor_graph->factor_node[i].marginal[a][b] > 1e-10)
                    h_e += - factor_graph->factor_node[i].marginal[a][b] * log(factor_graph->factor_node[i].marginal[a][b]);
            }
        Z += h_e;
    }
    // Node entroy
    for (int i = 0; i < n; i++)
    {
        double h_v = 0.0;
        for (int a = 0; a < num_label; a++)
            if (fabs(factor_graph->var_node[i].marginal[a]) > 1e-10)
                h_v += - factor_graph->var_node[i].marginal[a] * log(factor_graph->var_node[i].marginal[a]);
        Z -= h_v * ((int)factor_graph->var_node[i].neighbor.size() - 1);
    }
    
    f -= Z;

    return f;
}

void CRFModel::LBP_Test()
{
    factor_graph->ClearDataForMaxSum();
    factor_graph->labeled_given = true;

    for (int i = 0; i < num_node; i++)
    {
        double* p_lambda = lambda;
        for (int y = 0; y < num_label; y++)
        {
            if (data->node[i]->label_type == KNOWN_LABEL)
            {
                factor_graph->SetVariableStateFactor(i, y, (y == data->node[i]->label));
            }
            else 
            {
                double v = 1.0;
                for (int t = 0; t < data->node[i]->num_attrib; t ++)
                    v *= exp( p_lambda[ data->node[i]->attrib[t] ] * data->node[i]->value[t] );
                factor_graph->SetVariableStateFactor(i, y, v);
            }  
            p_lambda += num_attrib_type;
        }
    }    

    factor_graph->MaxSumPropagation(conf->max_infer_iter);

    int* inf_label = new int[num_node];

    for (int i = 0; i < num_node; i++)
    {
        int ybest = -1;
        double vbest, v;
        double vsum = 0.0;
        for (int y = 0; y < num_label; y++)
        {
            v = factor_graph->var_node[i].state_factor[y];
            for (int t = 0; t < factor_graph->var_node[i].neighbor.size(); t++)
                v *= factor_graph->var_node[i].belief[t][y];
            if (ybest < 0 || v > vbest)
                ybest = y, vbest = v;

            probability[i][y] = v;
            vsum += v;
        }

        inf_label[i] = ybest;

        for (int y = 0; y < num_label; y++)
            probability[i][y] /= vsum;
    }

    int hit = 0, miss = 0;
    int hitu = 0, missu = 0;

    for (int i = 0; i < num_node; i++)
    {
        if (inf_label[i] == data->node[i]->label)
            hit++;
        else
            miss++;

        if (data->node[i]->label_type != KNOWN_LABEL)
        {
            if (inf_label[i] == data->node[i]->label)
                hitu++;
            else
                missu++;
        }
    }

    printf("A_Acc = %.4lf  U_Acc = %.4lf\n", (double)hit/(hit+miss), (double)hitu/(hitu+missu));
    fflush(stdout);

    // FILE* fprob = fopen("uncertainty.txt", "w");
    // for (int i = 0; i < n; i++)
    // {
    //  if (data->node[i]->label_type == KNOWN_LABEL)
    //  {
    //      for (int y = 0; y < num_label; y++)
    //          fprintf(fprob, "%s -1 ", data->label_dict.GetKeyWithId(y).c_str());
    //      fprintf(fprob, "\n");
    //  }
    //  else
    //  {
    //      for (int y = 0; y < num_label; y++)
    //          fprintf(fprob, "%s %.4lf ", data->label_dict.GetKeyWithId(y).c_str(), label_prob[y][i]);
    //      fprintf(fprob, "\n");
    //  }
    // }
    // fclose(fprob);
    
    delete[] inf_label;
}

