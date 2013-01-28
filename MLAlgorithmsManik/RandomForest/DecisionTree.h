#ifndef __DECISIONTREE_H__
#define __DECISIONTREE_H__

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
#include <vector>
#include "Config.h"
#include "InstanceData.h"
#define NUM_MULTI_FEATURES 20

using std::ostream;
using std::pair;
using std::vector;

class SplitFeature
{
public:
	SplitFeature(const InstanceData& instance) : instanceData(instance)
	{ }
	
	virtual double Evaluate(InstanceID i) const
	{
		return 0.0;
	}
	virtual ~SplitFeature() { }
	friend ostream& operator<<(ostream & o, SplitFeature *obj);

	virtual string GetRepresentation()
	{
		return "";
	}
    static SplitFeature* Parse (string s)
	{
		return NULL;
	}

	virtual void insertFeature(FeatureID f){
	}


protected:
	const InstanceData &instanceData;
};

class MultipleFeature : public SplitFeature

{
public:
	MultipleFeature(const InstanceData& instance,int weight)

		: SplitFeature(instance), weight(weight)

	{ }

	virtual double Evaluate(InstanceID i) const

	{
		double p=0;
		for(int j=0;j<features.size();j++){
			int q=((weight & (1<<j)))?1:-1;

			FeatureID r=features[j];
			p+=this->instanceData.GetFeature(i,r)*q;
		}
		return p;
	}	

	~MultipleFeature() {}

	virtual void insertFeature(FeatureID f){
		this->features.push_back(f);
	}

		std::vector<FeatureID> features;
		int weight;
};

class SingleFeature : public SplitFeature
{
public:
	SingleFeature(const InstanceData& instance, FeatureID f)
		: SplitFeature(instance), featureID(f)
	{ }
	virtual double Evaluate( InstanceID i ) const
	{
		return this->instanceData.GetFeature(i, this->featureID);
	}

	~SingleFeature() { }
	virtual string GetRepresentation()
	{
		std::stringstream out;
		out << "S" << featureID;
		return out.str();
	}
    static SingleFeature* Parse (string s, const InstanceData &data)
	{
		return new SingleFeature(data, atoi(s.substr(1).c_str()));
	}
private:
	FeatureID featureID;
};



class RandomCombination : public SplitFeature
{
};

class CandidateSplit
{
public:
	CandidateSplit(double evaluation, unsigned int leftInstances, SplitFeature *feature, double splitValue)
		: _leftInstances(leftInstances), _feature(feature), _evaluation(evaluation), _splitValue(splitValue)
	{ }

	unsigned int _leftInstances;
	SplitFeature *_feature;
	double _splitValue;
	double _evaluation;
};

class Node
{
public:
	Node (unsigned int id, unsigned int l, unsigned int r, vector<InstanceID> &i,
		    unsigned int lvl,
			unsigned int m = static_cast<unsigned int>(-1),
			bool s=false, SplitFeature *f=NULL, double svalue=0.0)
		: Id(id), left(l), right(r), level(lvl), mid(m), instances(i), splitted(s), feature(f), splitValue(svalue)
	{ }
	unsigned int Id;
	unsigned int left;
	unsigned int right;
	unsigned int mid;
	unsigned int level;
	bool         splitted;
	vector<InstanceID> &instances;
	SplitFeature *feature;
	double splitValue;
};

class DecisionTree
{
public:
	DecisionTree (const InstanceData& instances, Config cfg, string Id, unsigned int seed)
		: instanceData(instances), config(cfg), id(Id), featureSampler(0, instances.NumFeatures()-1)//, engine(rd())
	{

		
		gen.seed(time(NULL) * seed);

		// srand(time(NULL) * seed * seed);
		// std::cout << seed << "\t" << rand() << std::endl;
	}
	void Grow ( vector <InstanceID> &instances );
	void DumpTree (std::ostream &o);
	~DecisionTree()
	{
		for(vector<SplitFeature *>::iterator it = _splitFeatures.begin();
			it != _splitFeatures.end(); it++)
		{
			delete (*it);
		}
	}

protected:
	const InstanceData &instanceData;
	Config config;
	Node ExpandNode (Node);
	vector <SplitFeature *> *CandidateSplitFeature( const Node & );
	CandidateSplit EvaluateFeature (const Node &n, SplitFeature *f);
	double AverageProbability(unsigned int begin, unsigned int end, vector<pair<double, InstanceID>> &values);
	double ConditionalProbabilty(unsigned int begin, unsigned int end, vector<pair<double, InstanceID>> &values);
	map<unsigned int, double>
		GetConditionalProbabilty(unsigned int begin, unsigned int end, vector<pair<double, InstanceID>> &values);
	map<unsigned int, double>
		GetAverageProbabilty(unsigned int begin, unsigned int end, vector<pair<double, InstanceID>> &values);
	double DecisionTree::Entropy(unsigned int begin, unsigned int end,
		vector<pair<double, InstanceID>> &values,
		map<unsigned int, double> (DecisionTree::*funcptr) (unsigned int begin, unsigned int end,
		vector<pair<double, InstanceID>> &values));
	double DecisionTree::Gini(unsigned int begin, unsigned int end,
		vector<pair<double, InstanceID>> &values,
		map<unsigned int, double> (DecisionTree::*funcptr) (unsigned int begin, unsigned int end,
		vector<pair<double, InstanceID>> &values));
	double DecisionTree::Twoing(unsigned int begin, unsigned int end, unsigned int middle,
		vector<pair<double, InstanceID>> &values,
		map<unsigned int, double> (DecisionTree::*funcptr) (unsigned int begin, unsigned int end,
		vector<pair<double, InstanceID>> &values));
	double DecisionTree::Orthogonal(unsigned int begin, unsigned int end, unsigned int middle,
		vector<pair<double, InstanceID>> &values,
		map<unsigned int, double> (DecisionTree::*funcptr) (unsigned int begin, unsigned int end,
		vector<pair<double, InstanceID>> &values));

	double Variance(SplitFeature *f, const Node &n);
	FeatureID GetRandomFeature ( void );
	void DecisionTree::AddProbability(double *v,double* u,unsigned int begin, unsigned int end, vector<pair<double, InstanceID>> &values);

	vector <SplitFeature *> _splitFeatures;
	vector <double>         _splitThresholds;
	vector <unsigned int>   leftChild;
	vector <unsigned int>   rightChild;
	vector <map<LabelID, double>>   labelDist;
	string id;
	time_t start, end;
    std::tr1::uniform_int<FeatureID> featureSampler;
		// std::random_device rd;
		// std::mt19937 engine;
	std::tr1::minstd_rand gen;
};


#endif