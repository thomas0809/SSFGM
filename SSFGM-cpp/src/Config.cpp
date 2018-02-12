#include "Config.h"
#include <cstdlib>
#include <cstring>

void Config::SetDefault()
{
    max_iter = 50;
    max_infer_iter = 10;

    this->task = "-est";
    this->method = "LBP";
    this->train_file = "example.txt";

    this->pred_file = "pred.txt";

    this->has_attrib_value = true;
    this->directed = false;
    this->eps = 1e-3;

    this->gradient_step = 0.001;

    this->src_model_file = "";
    this->dst_model_file = "model.txt";

    this->eval_interval = 10;

    this->num_thread = 1;

    this->batch_size = 1000;
    this->early_stop_patience = 10;
    this->state_file = "";
}

bool Config::LoadConfig(int argc, char* argv[])
{
    if (argc == 1) return 0;

    int i = 1;
    if (strcmp(argv[1], "-est") == 0 || strcmp(argv[1], "-estc") == 0 || strcmp(argv[1], "-inf") == 0)
    {
        this->task = argv[1];
        i++;
    }
    else return 0;

    while (i < argc)
    {
        if (strcmp(argv[i], "-method") == 0)
            this->method = argv[++i]; 

        else if (strcmp(argv[i], "-trainfile") == 0)
            this->train_file = argv[++i];
        
        else if (strcmp(argv[i], "-srcmodel") == 0)
            this->src_model_file = argv[++i];
        
        else if (strcmp(argv[i], "-dstmodel") == 0)
            this->dst_model_file = argv[++i]; 

        else if (strcmp(argv[i], "-pred") == 0)
            this->pred_file = argv[++i];

        else if (strcmp(argv[i], "-state") == 0)
            this->state_file = argv[++i]; 

        else if (strcmp(argv[i], "-niter") == 0)
            this->max_iter = atoi(argv[++i]);

        else if (strcmp(argv[i], "-ninferiter") == 0)
            this->max_infer_iter = atoi(argv[++i]);

        else if (strcmp(argv[i], "-lrate") == 0)
            this->gradient_step = atof(argv[++i]);

        else if (strcmp(argv[i], "-hasvalue") == 0)
            this->has_attrib_value = true; 
        
        else if (strcmp(argv[i], "-novalue") == 0)
            this->has_attrib_value = false; 

        else if (strcmp(argv[i], "-directed") == 0)
            this->directed = true; 

        else if (strcmp(argv[i], "-eval") == 0)
            this->eval_interval = atoi(argv[++i]); 

        else if (strcmp(argv[i], "-earlystop") == 0)
            this->early_stop_patience = atoi(argv[++i]); 

        else if (strcmp(argv[i], "-batch") == 0)
            this->batch_size = atoi(argv[++i]); 

        else if (strcmp(argv[i], "-thread") == 0)
            this->num_thread = atoi(argv[++i]);
        
        i++;
    }
    return 1;
}

void Config::ShowUsage()
{
    printf("Semi-Supervised Factor Graph Model                           \n");
    //printf("     by Wenbin Tang, Tsinghua University                     \n");
    printf("                                                             \n");
    printf("Usage: mpiexec -n NUM_PROCS OpenCRF <task> [options]         \n");
    printf(" Options:                                                    \n");
    printf("   task: -est, -estc, -inf                                   \n");    
    printf("\n");
    printf("   -niter int           : number of iterations                               \n");
    printf("   -ninferiter int      : number of iterations for inference                 \n");
    printf("   -srcmodel string     : the model to load                                  \n");
    printf("   -dstmodel string     : model file to save                                 \n");
    printf("   -method string       : methods (LBP/MH/MH1), default: LBP                 \n");
    printf("   -lrate double        : learning rate                                      \n");
    printf("   -hasvalue            : [default] attributes with values (format: attr:val)\n");
    printf("   -novalue             : attributes without values (format: attr)           \n");
    printf("   -trainfile string    : train file                                         \n");
    printf("   -testfile string     : test file                                          \n");
    printf("   -neval int           : interval of evaluation                             \n");
    printf("   -earlystop int       : early stop patience                                \n");
    printf("   -batch int           : batch size                                         \n");
    printf("   -state string        : state file                                         \n");

    exit(0);
}
