#include <ppl.h>
#include <string>
#include <vector>

#include <iostream>
#include <sstream>
#include <random>

#include "Config.h"
#include "DecisionTree.h"
#include "InstanceData.h"

using namespace std;
using std::string;
using std::vector;

int main(int argc, char*argv[])
{
	//for(int i=0;i<10;i++){

	// Load the data
	string featureFile(argv[1]);
	string labelFile(argv[2]);

	InstanceData data(featureFile, labelFile);
	
	Config cfg;
	if (string(argv[4]) == "T")
		cfg.eLambda = true;
	else
		cfg.eLambda  = false;

	if (atoi(argv[5]) == 1)
		cfg.features = cfg.SingleFeature;
	else
		cfg.features = cfg.RandomCombination;

	cfg.lambda				= atof(argv[6]);
	cfg.maxDepth			= atoi(argv[7]);
	cfg.minInstanceAtLeaf	= atoi(argv[8]);
	cfg.numFeatures			= atoi(argv[9]);
	if (atoi(argv[10]) == cfg.AverageProbability)
		cfg.probability = cfg.AverageProbability;
	else if (atoi(argv[10]) == cfg.ConditionalProbability)
		cfg.probability = cfg.ConditionalProbability;
	else
		return -1;
	switch(atoi(argv[11]))
	{
	case cfg.Entropy:
		cfg.criteria = cfg.Entropy;
		break;
	case cfg.GiniIndex:
		cfg.criteria = cfg.GiniIndex;
		break;
	case cfg.Twoing:
		cfg.criteria = cfg.Twoing;
		break;
	case cfg.Orthogonal:
		cfg.criteria = cfg.Orthogonal;
		break;

	default:
		return -1;
	}
	cfg.varianceThreshold   = atof(argv[12]);
	cfg.numTrees            = atoi(argv[13]);
	cfg.maxSplits           = atof(argv[14]);

	std::ofstream dTreeFile(argv[3]);


	cfg.Display(std::cout);
	cfg.Display(dTreeFile);

	/*
	vector<InstanceID> instances;
	std::cout << instances.size() << std::endl;
	for (unsigned int i = 0; i < data.NumInstances(); i++)
		instances.push_back(i);
	*/

	vector<vector<InstanceID>> instance_array;

	/*
	for(unsigned int i = 0; i < cfg.numTrees; i++)
	{
		dTreeFile << "TimeStarted:" << time(NULL) << std::endl;
		std::stringstream out;
		out << i;
		DecisionTree dt(data, cfg, out.str());
		dt.Grow(instances);
		dt.DumpTree(dTreeFile);
		dTreeFile << "TimeStopped:" << time(NULL) << std::endl;
	}

	*/
	vector<DecisionTree> forest;
	std::tr1::uniform_int<InstanceID> instanceSampler(0, data.NumInstances()-1);
	std::tr1::minstd_rand gen;

	for(unsigned int i = 0; i < cfg.numTrees; i++)
	{
		std::stringstream out;
		out << i;
		dTreeFile << "TimeStarted\t" << i << "\t" << time(NULL) << std::endl;
		forest.push_back(DecisionTree(data, cfg, out.str(),i + 1));

		vector<InstanceID> instances;
		for (unsigned int i = 0; i < data.NumInstances(); i++)
		{
			// std::cout << instanceSampler(gen) << std::endl;
			instances.push_back(instanceSampler(gen) % data.NumInstances());
		}
		instance_array.push_back(instances);
		std::cerr << "STATS " << instances.size() << std::endl;
	}
	// std::ofstream timings("C:\\RFData\\CSVZips\\Csv-Format\\MEDIAMILL\\timings.txt", std::ios::out | std::ios::app);
	// timings << "TimeStarted\t" << i << "\t" << time(NULL) << "\t";

	
	Concurrency::parallel_for(0u, cfg.numTrees, [&forest, &instance_array](size_t i)
	{
		forest[i].Grow(instance_array[i]);
	});
	

	// forest[0].Grow(instance_array[0]);
	// timings << "TimeEnded  \t" << i << "\t" << time(NULL) << std::endl;
	std::cout<<"\n\n\n\n\n\nFinally Over\n\n\n\n\n\nNow writing to disk";
	// dump the trees here
	for(unsigned int i = 0; i < cfg.numTrees; i++){
		std::cout<<i<<std::endl;
		forest[i].DumpTree(dTreeFile);
	}
	return 0;
}