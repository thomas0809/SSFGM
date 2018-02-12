#include "Config.h"
#include "DataSet.h"
#include "CRFModel.h"

int main(int argc, char* argv[])
{
	// Load Configuartion
	Config* conf = new Config();
	if (! conf->LoadConfig(argc, argv))
	{
		conf->ShowUsage();
		return 0;
	}
	printf("Load data...\n");
	
	CRFModel *model = new CRFModel(conf);

	if (conf->task == "-est")
	{
		model->Estimate();
	}
	else if (conf->task == "-inf")
	{
		model->Inference();
	}
	else
	{
		Config::ShowUsage();
	}

	return 0;
}