#include <random>
#include <ctime>
#include <set>

#include "CRFModel.h"
#include "Constant.h"

using namespace std;

#define MAX_BUF_SIZE 65536

void CRFModel::MH_Init()
{
    state.clear();
    state.assign(num_node, 0);
    for (int i = 0; i < num_node; i++)
    {
        double maxsum = 0;
        for (int y = 0; y < num_label; y++)
        {
            double sum = 0;
            for (int j = 0; j < data->node[i]->num_attrib; j++)
                sum += lambda[GetAttribParameterId(y, data->node[i]->attrib[j])] * data->node[i]->value[j];
            if (sum > maxsum || y == 0)
            {
                maxsum = sum;
                state[i] = y;
            }
        }
        if (data->node[i]->label_type == KNOWN_LABEL)
            state[i] = data->node[i]->label;
    }
}

void CRFModel::MH_Train()
{
    int N = num_node;

    int max_iter         = conf->max_iter; 
    int batch_size       = conf->batch_size;
    int max_infer_iter   = conf->max_infer_iter; 
    double learning_rate = conf->gradient_step;

    int num_thread       = conf->num_thread;

    map<int,double>* gradient_thread = new map<int,double>[num_thread];
    vector<int>* state_thread = new vector<int>[num_thread];
    for (int i = 0; i < num_thread; i++)
        state_thread[i] = state;

    double best_valid_acc = -1;
    int valid_wait = 0;
    double *best_lambda = new double[num_parameter];
    memcpy(best_lambda, lambda, num_parameter * sizeof(double));


    for (int iter = 0; iter < max_iter; iter += conf->eval_interval)
    {
        if (iter % conf->eval_interval == 0)
        {
            printf("[Iter %d]", iter);
            state = state_thread[0];
            double valid_acc = MH_Test(0);
            if (valid_acc > best_valid_acc)
            {
                memcpy(best_lambda, lambda, num_parameter * sizeof(double));
                best_valid_acc = valid_acc;
                valid_wait = 0;
                for (int thread_id = 0; thread_id < num_thread; thread_id++)
                    state_thread[thread_id] = state;
            }
            else
            {
                valid_wait++;
                if (valid_wait > conf->early_stop_patience)
                    break;
            }
        }
        int update = 0;

        #pragma omp parallel for num_threads(num_thread)
        for (int thread_id = 0; thread_id < num_thread; thread_id++) 
        {
            vector<int>& _state = state_thread[thread_id];
            map<int,double>& _gradient = gradient_thread[thread_id];

            random_device rd;
            static thread_local mt19937 gen(rd());
            uniform_int_distribution<int> rand_N(0, N - 1);
            uniform_int_distribution<int> rand_CLASS(0, num_label - 1);
            uniform_real_distribution<double> rand_P(0, 1);

            for (int iter_thread = 0; iter_thread < conf->eval_interval; iter_thread++)
            {
                _gradient.clear();
                //int iters = (batch_size + thread_id) / num_thread;
                for (int batch_iter = 0; batch_iter < batch_size; batch_iter++)
                {
                    map<int,int> change;
                    map<int,double> gradient1, gradient2;

                    int acc1 = 0, acc2 = 0;
                    double likeli1 = 0, likeli2 = 0;
                    
                    // generate change set
                    int center = rand_N(gen);
                    change[center] = _state[center];

                    // calculate for Y
                    map<int,int>::iterator it;
                    for (it = change.begin(); it != change.end(); it++)
                    {
                        int u = it->first;
                        acc1 += (data->node[u]->label_type == KNOWN_LABEL && _state[u] == data->node[u]->label);
                        likeli1 += CalcLikelihood(u, _state, &gradient1);
                    }

                    // change Y to Ynew
                    for (it = change.begin(); it != change.end(); it++)
                        _state[it->first] = rand_CLASS(gen);

                    // calculate for Ynew
                    for (it = change.begin(); it != change.end(); it++)
                    {
                        int u = it->first;
                        acc2 += (data->node[u]->label_type == KNOWN_LABEL && _state[u] == data->node[u]->label);
                        likeli2 += CalcLikelihood(u, _state, &gradient2);
                    }

                    // accept/reject Ynew
                    double accept = min(1.0, exp(likeli2 - likeli1));
                    double p = rand_P(gen);
                    if (p > accept) // reject
                    {
                        for (it = change.begin(); it != change.end(); it++)
                            _state[it->first] = it->second;
                    }
                    else // update lambda
                    {
                        double step = 0;
                        if (acc2 > acc1 && likeli2 <= likeli1)
                            step = 1;
                        else if (acc2 < acc1 && likeli2 > likeli1)
                            step = -1;
                        if (step != 0)
                        {
                            update += 1;
                            for (map<int,double>::iterator i = gradient1.begin(); i != gradient1.end(); i++)
                            {
                                int id = i->first; 
                                double val = i->second;
                                _gradient[id] = _gradient[id] - step * val;
                            }
                            for (map<int,double>::iterator i = gradient2.begin(); i != gradient2.end(); i++)
                            {
                                int id = i->first;
                                double val = i->second;
                                _gradient[id] = _gradient[id] + step * val;
                            }
                        }
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
    state = state_thread[0];
    MH_Test(max_infer_iter);
}

double CRFModel::MH_Test(int max_iter)
{
    int N = num_node;
    int num_thread = conf->num_thread;

    for (int i = 0; i < N; i++)
        if (data->node[i]->label_type == KNOWN_LABEL)
            state[i] = data->node[i]->label;
    
    vector<int> &best_state = state;
    double best_likeli = 0;

    printf("EVAL#"); 
    fflush(stdout);

    #pragma omp parallel for num_threads(num_thread)
    for (int thread_id = 0; thread_id < num_thread; thread_id ++) 
    {
        std::random_device rd;
        std::default_random_engine gen(time(0));
        std::uniform_int_distribution<int> rand_U(0, unlabeled.size() - 1);
        std::uniform_int_distribution<int> rand_CLASS(0, num_label - 1);
        std::uniform_real_distribution<double> rand_P(0, 1);

        vector<int> _state = state;
        vector<int> _best_state;
        double _best_likeli = 0;

        map<int,int> change;
        double _state_likeli = 0;

        int iters = (max_iter + thread_id) / num_thread;
        for (int iter = 0; iter < iters; iter++)
        {
            change.clear();
            double likeli1 = 0, likeli2 = 0;

            // generate change set
            int center = unlabeled[rand_U(gen)];
            if (data->node[center]->label_type == KNOWN_LABEL)
                continue;
            change[center] = _state[center];

            // calculate for Y
            map<int,int>::iterator it;
            for (it = change.begin(); it != change.end(); it++)
                likeli1 += CalcLikelihood(it->first, _state, NULL);

            // change Y to Ynew
            for (it = change.begin(); it != change.end(); it++)
                _state[it->first] = rand_CLASS(gen);

            // calculate for Ynew
            for (it = change.begin(); it != change.end(); it++)
                likeli2 += CalcLikelihood(it->first, _state, NULL);

            // accept/reject Ynew
            double accept = min(1.0, exp(likeli2 - likeli1));
            double p = rand_P(gen);
            if (p > accept)  // reject
            { 
                for (it = change.begin(); it != change.end(); it++)
                    _state[it->first] = it->second;
            }
            else
            {
                _state_likeli = _state_likeli + likeli2 - likeli1;
            }
            if (_state_likeli > _best_likeli)
            {
                _best_state = _state;
                _best_likeli = _state_likeli;
            } 
        }
        #pragma omp critical
        {
            if (_best_likeli > best_likeli) {
                best_likeli = _best_likeli;
                best_state = _best_state;
            }
        }
    }

    set<int> candidate;
    int changed;
    for (int i = 0; i < unlabeled.size(); i++)
        candidate.insert(unlabeled[i]);

    for (int iter = 0; iter < 2; iter++)
    {
        printf("#"); fflush(stdout);

        int vhit = 0, vall = 0;
        for (int i = 0; i < valid.size(); i++)
        {
            int u = valid[i];
            vall += 1;
            vhit += (data->node[u]->label == best_state[u]);   
        }
        double valid_auc = double(vhit) / double(max(vall, 1));
        printf("%.4f ", valid_auc);
        fflush(stdout);

        set<int> new_candidate;
        changed = 0;
        for (set<int>::iterator i = candidate.begin(); i != candidate.end(); i++)
        {
            int u = *i;
            int yold = best_state[u];
            double *p = probability[u];
            for (int k = 0; k < num_label; k++)
            {
                best_state[u] = k;
                p[k] = CalcLikelihood(u, best_state, NULL);
            }
            int ynew = 0;
            for (int k = 0; k < num_label; k++)
            {
                if (p[k] > p[ynew])
                    ynew = k;
            }
            best_state[u] = ynew;
            if (ynew != yold) {
                changed += 1;
                new_candidate.insert(u);
                for (int j = 0; j < data->graph->outlist[u].size(); j++) {
                    int v = data->graph->outlist[u][j].first;
                    if (data->node[v]->label_type == KNOWN_LABEL)
                        continue;
                    new_candidate.insert(v);
                }
                for (int j = 0; j < data->graph->inlist[u].size(); j++) {
                    int v = data->graph->inlist[u][j].first;
                    if (data->node[v]->label_type == KNOWN_LABEL)
                        continue;
                    new_candidate.insert(v);
                }
                // printf("%d %d %f %f\n", u, data->node[u]->label, p[data->node[u]->label], p[ynew]);
            }
        }
        printf("%d", changed);
        candidate = new_candidate;
        if (candidate.size() == 0)
            continue;
    }

    int hit = 0, all = 0;
    for (int i = 0; i < test.size(); i++)
    {
        int u =  test[i];
        all += 1;
        hit += (data->node[u]->label == best_state[u]);
    }
    int vhit = 0, vall = 0;
    for (int i = 0; i < valid.size(); i++)
    {
        int u = valid[i];
        vall += 1;
        vhit += (data->node[u]->label == best_state[u]);   
    }
    double valid_auc = double(vhit) / double(max(vall, 1));
    printf(" Accuracy: %d/%d = %.4f, Valid: %.4f\n", hit, all, double(hit) / double(all), valid_auc);
    fflush(stdout);
    return valid_auc;
}


double CRFModel::CalcLikelihood(int u, vector<int> &_state, map<int,double> *gradient)
{
    double likeli = 0;
    for (int j = 0; j < data->node[u]->num_attrib; j++)
    {
        int pid = GetAttribParameterId(_state[u], data->node[u]->attrib[j]);
        double val = data->node[u]->value[j];
        likeli += lambda[pid] * val;
        if (gradient != NULL)
            (*gradient)[pid] = (*gradient)[pid] + val;
    }
    for (int j = 0; j < data->graph->outlist[u].size(); j++) 
    {
        int v = data->graph->outlist[u][j].first;
        int edge_type = data->graph->outlist[u][j].second;
        int pid = GetEdgeParameterId(edge_type, _state[u], _state[v]);
        likeli += lambda[pid];
        if (gradient != NULL)
            (*gradient)[pid] = (*gradient)[pid] + 1;
    }
    for (int j = 0; j < data->graph->inlist[u].size(); j++) 
    {
        int v = data->graph->inlist[u][j].first;
        int edge_type = data->graph->inlist[u][j].second;
        int pid = GetEdgeParameterId(edge_type, _state[v], _state[u]);
        likeli += lambda[pid];
        if (gradient != NULL)
            (*gradient)[pid] = (*gradient)[pid] + 1;
    }
    return likeli;
}

