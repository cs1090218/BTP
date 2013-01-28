#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <fstream>
#include <iostream>

struct Config
{
	enum Probability
	{
		ConditionalProbability = 2, AverageProbability = 1
	};
	enum Features
	{
		SingleFeature, RandomCombination
	};

	enum Criteria
	{
		Entropy=1, GiniIndex, Twoing, Orthogonal
	};

	unsigned int minInstanceAtLeaf;
	unsigned int maxDepth;
	bool         eLambda;
	double       lambda;
	Probability  probability;
	Features     features;
	Criteria     criteria;
	unsigned int numFeatures;
	double       varianceThreshold;
	unsigned int numTrees;
	double       maxSplits;
	void Display ( std::ostream &o )
	{
		o << "minInstanceAtLeaf : " << minInstanceAtLeaf	<< std::endl
			<< "maxDepth : "				<< maxDepth				<< std::endl
			<< "eLambda : "					<< eLambda				<< std::endl
			<< "lambda : "					<< lambda				<< std::endl
			<< "probability : "				<< probability			<< std::endl
			<< "features : "				<< features				<< std::endl
			<< "numFeatures : "				<< numFeatures			<< std::endl
			<< "varianceThreshold : "       << varianceThreshold    << std::endl
			<< "maxSplits : "               << maxSplits            << std::endl;
	}
};

#endif