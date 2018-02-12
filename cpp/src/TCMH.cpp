#include "CRFModel.h"
#include "Constant.h"

#include <random>
#include <ctime>

using namespace std;

void CRFModel::TCMH_Train()
{
    int max_iter         = conf->max_iter;
    int batch_size       = conf->batch_size;
    int max_infer_iter   = conf->max_infer_iter;
    double learning_rate = conf->gradient_step;
    int num_thread       = conf->num_thread;

    int N = num_node;

    // map<int, double> sum1, sum2;
    
    vector<int>* state1_thread = new vector<int>[num_thread];
    vector<int>* state2_thread = new vector<int>[num_thread];
    for (int i = 0; i < num_thread; i++) 
    {
        state1_thread[i] = state;
        state2_thread[i] = state;
    }

    double best_valid_acc = -1;
    int valid_wait = 0;
    double *best_lambda = new double[num_parameter];
    memcpy(best_lambda, lambda, num_parameter * sizeof(double));

    for (int iter = 0; iter < max_iter; iter += conf->eval_interval)
    {
        if (iter % conf->eval_interval == 0)
        {
            printf("[Iter %d]", iter);
            state = state1_thread[0];
            double valid_acc = MH_Test(0);
            if (valid_acc > best_valid_acc)
            {
                memcpy(best_lambda, lambda, num_parameter * sizeof(double));
                best_valid_acc = valid_acc;
                valid_wait = 0;
                for (int thread_id = 0; thread_id < num_thread; thread_id++)
                {
                    state1_thread[thread_id] = state;
                    state2_thread[thread_id] = state;
                }
            }
            else
            {
                if (iter / conf->eval_interval >= 20)
                    valid_wait++;
                if (valid_wait > conf->early_stop_patience)
                    break;
            }
        }

        #pragma omp parallel for num_threads(num_thread)
        for (int thread_id = 0; thread_id < num_thread; thread_id++) 
        {
            random_device rd;
            static thread_local mt19937 gen(rd());
            uniform_int_distribution<int> rand_N(0, N - 1);
            uniform_int_distribution<int> rand_CLASS(0, num_label - 1);
            uniform_real_distribution<double> rand_P(0, 1);

            map<int, double> _gradient1, _gradient2;
            map<int, double> _gradient;
            vector<int> &_state1 = state1_thread[thread_id];
            vector<int> &_state2 = state2_thread[thread_id];
            
            // int iters = (batch_size + thread_id) / num_thread;
            for (int iter_thread = 0; iter_thread < conf->eval_interval; iter_thread++)
            {
                _gradient.clear();
                for (int it = 0; it < batch_size; it++)
                {
                    int center = rand_N(gen);
                    int y = rand_CLASS(gen);
                    double p = rand_P(gen);
                    bool b1 = train1_sample(1, center, y, p, _state1, _gradient1);
                    bool b2 = train1_sample(2, center, y, p, _state2, _gradient2);
                    if (!b1 && !b2)
                        continue;
                    map<int,double>::iterator itt;
                    for (itt = _gradient1.begin(); itt != _gradient1.end(); itt++)
                    {
                        int pid = itt->first;
                        double val = itt->second;
                        _gradient[pid] = _gradient[pid] + val;
                    } 
                    for (itt = _gradient2.begin(); itt != _gradient2.end(); itt++)
                    {
                        int pid = itt->first;
                        double val = itt->second;
                        _gradient[pid] = _gradient[pid] - val;
                    }
                }
                double norm = 1; //double(batch_size);
                for (map<int,double>::iterator i = _gradient.begin(); i != _gradient.end(); i++)
                {
                    int id = i->first;
                    double val = i->second;
                    lambda[id] += learning_rate * val / norm;
                }
            }
        }
    }

    memcpy(lambda, best_lambda, num_parameter * sizeof(double));
    state = state1_thread[0];
    MH_Test(max_infer_iter);
}

bool CRFModel::train1_sample(int type, int center, int ynew, double p, vector<int>& _state, map<int,double>& _gradient)
{
    int u = center;
    int y_center = _state[center];
    double likeli = 0, likeli1 = 0;
    map<int,double> temp, temp1;
    temp.clear(); temp1.clear();

    // calculate for Y
    for (int j = 0; j < data->node[u]->num_attrib; j++)
    {
        int pid = GetAttribParameterId(_state[u], data->node[u]->attrib[j]);
        double val = data->node[u]->value[j];
        likeli += lambda[pid] * val;
        temp[pid] = temp[pid] + val;
    }
    for (int j = 0; j < data->graph->outlist[u].size(); j++) 
    {
        int v = data->graph->outlist[u][j].first;
        int edge_type = data->graph->outlist[u][j].second;
        int pid = GetEdgeParameterId(edge_type, _state[u], _state[v]);
        likeli += lambda[pid];
        temp[pid] = temp[pid] + 0.5;
    }
    for (int j = 0; j < data->graph->inlist[u].size(); j++) {
        int v = data->graph->inlist[u][j].first;
        int edge_type = data->graph->inlist[u][j].second;
        int pid = GetEdgeParameterId(edge_type, _state[v], _state[u]);
        likeli += lambda[pid];
        temp[pid] = temp[pid] + 0.5;
    }

    if (type == 1 && data->node[center]->label_type == KNOWN_LABEL)
    {
        _gradient = temp;
        return false;
    }

    // change Y to Ynew
    _state[center] = ynew;

    // calculate for Ynew
    for (int j = 0; j < data->node[u]->num_attrib; j++) 
    {
        int pid = GetAttribParameterId(_state[u], data->node[u]->attrib[j]);
        double val = data->node[u]->value[j];
        likeli1 += lambda[pid] * val;
        temp1[pid] = temp1[pid] + val;
    }
    for (int j = 0; j < data->graph->outlist[u].size(); j++) 
    {
        int v = data->graph->outlist[u][j].first;
        int edge_type = data->graph->outlist[u][j].second;
        int pid = GetEdgeParameterId(edge_type, _state[u], _state[v]);
        likeli1 += lambda[pid];
        temp1[pid] = temp1[pid] + 0.5;
    }
    for (int j = 0; j < data->graph->inlist[u].size(); j++) 
    {
        int v = data->graph->inlist[u][j].first;
        int edge_type = data->graph->inlist[u][j].second;
        int pid = GetEdgeParameterId(edge_type, _state[v], _state[u]);
        likeli1 += lambda[pid];
        temp1[pid] = temp1[pid] + 0.5;
    }

    double accept = min(1.0, exp(likeli1 - likeli));
    if (p > accept) // reject
    {
        _state[center] = y_center;
        _gradient = temp;
        return false;
    }
    else // accept
    {
        _gradient = temp1;
        return true;
    }
}