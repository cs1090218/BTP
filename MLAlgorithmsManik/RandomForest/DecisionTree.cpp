#include <cfloat>
#include <cmath>
#include <map>
#include <queue>
#include <utility>
#include "DecisionTree.h"

#include <iostream>

using std::map;
using std::pair;
using std::queue;

ostream & operator<<(ostream &o, SplitFeature *obj)
{
	o << obj->GetRepresentation();
	return o;
}


void DecisionTree::Grow( vector<InstanceID> &instances)
{
	queue<Node> _nodes;
	unsigned int done = 0;
	this->start = time(NULL);

	// insert the root node
	unsigned idValue = 0;
	_nodes.push(Node(idValue, 0, instances.size()-1, instances, 0));
	idValue ++;
	unsigned int consumed = 0;
	while (!_nodes.empty())
	{
		// std::cout << "Going for: " << _nodes.front().left << " " << _nodes.front().right << std::endl;
		if(consumed % 100 == 0)
			std::cout << this->id << " Remaining: " << _nodes.size() << " done  " << done << " of " << this->instanceData.NumInstances() << std::endl;
		Node n = this->ExpandNode(_nodes.front());
		consumed ++;
		if (n.splitted)
		{
			this->_splitFeatures.push_back(n.feature);
			this->_splitThresholds.push_back(n.splitValue);
			this->leftChild.push_back(idValue);
			_nodes.push(Node(idValue, n.left, n.mid, n.instances, n.level+1));
			idValue ++;
			this->rightChild.push_back(idValue);
			_nodes.push(Node(idValue, n.mid + 1, n.right, n.instances, n.level+1));
			idValue ++;
		}
		else
		{
			this->_splitFeatures.push_back(NULL);
			this->_splitThresholds.push_back(0.0);
			this->leftChild.push_back(-1);
			this->rightChild.push_back(-1);
			done += (n.right + 1 - n.left);
		}

		// insert the label distribution
		this->labelDist.push_back(map<LabelID, double>());
		map<LabelID, double> &dist = this->labelDist[n.Id];

		for (unsigned int j = n.left; j <= n.right; j++)
		{
			const map<LabelID, double> &lbls = this->instanceData.Labels(n.instances[j]);
			for (map<LabelID, double>::const_iterator it = lbls.cbegin(); it != lbls.cend(); it++)
				dist[(*it).first] += (*it).second;
		}

		_nodes.pop();
	}
	this->end = time(NULL);
}

Node DecisionTree::ExpandNode( Node n)
{
	// minimum instance at a leaf
	if ( this->config.minInstanceAtLeaf >= (n.right - n.left + 1))
		return n;

	// maximum depth
	if ( this->config.maxDepth < n.level)
		return n;

	// generate candidates
	vector <SplitFeature *> *candidates = this->CandidateSplitFeature(n);
	CandidateSplit bestSplit(-DBL_MAX, 0, static_cast<SplitFeature *>(0), 0.0);
	for ( vector<SplitFeature *>::iterator it = candidates->begin();
		it != candidates->end(); it++)
	{
		CandidateSplit t = this->EvaluateFeature(n, *it);
		if (t._evaluation > bestSplit._evaluation)
			bestSplit = t;
	}

	// return appropriate splitInfo
	vector<InstanceID> tVector;
	if (bestSplit._feature)
	{
		tVector.resize(n.right - n.left + 1);
		for ( unsigned int i = n.left, l=0, r=bestSplit._leftInstances; i <= n.right; i++)
		{
			if ( bestSplit._feature->Evaluate(n.instances[i]) <= bestSplit._splitValue )
			{
				tVector[l] = n.instances[i];
				l++;
			}

			else
			{
				// std::cout << "r" << r << std::endl;
				tVector[r] = n.instances[i];
				r++;
			}
		}

		for (unsigned int i = n.left; i <= n.right; i++)
			n.instances[i] = tVector[i - n.left];
	}
	//std::cout<<"hi"<<std::endl;
	// free the candidate allocations
	for(vector<SplitFeature *>::iterator it = candidates->begin(); it != candidates->end(); it++)
		if (*it != bestSplit._feature)
			delete *it;
	
	delete candidates;

	if (bestSplit._feature)
		return Node(n.Id, n.left, n.right, n.instances, n.level, n.left + bestSplit._leftInstances - 1, true, bestSplit._feature, bestSplit._splitValue);
	else
		return n;
}

CandidateSplit DecisionTree::EvaluateFeature(const Node &n, SplitFeature *f)
{
	double totalInstances = n.right + 1 - n.left;
	vector<pair<double, InstanceID>> values (n.right + 1 - n.left);
	set<double> tValues;

	map<unsigned int, double> (DecisionTree::*funcptr) (unsigned int begin, unsigned int end,
	vector<pair<double, InstanceID>> &values);

	// std::cout << time(NULL) << " Start " << this->id << std::endl;

	for( unsigned int i = n.left, j = 0; i <= n.right; i++, j++ )
	{
		// std::cout << n.instances[i] << std::endl;
		double val = f->Evaluate(n.instances[i]);
		values[j] = pair<double, InstanceID> (val, n.instances[i]);
		tValues.insert(val);
	}

	
	// generate the splits if we are more than the maxSplits

	set<double> sValues;

	if (this->config.maxSplits < tValues.size())
	{
		vector<double> tList;
		for(set<double>::iterator it = tValues.begin(); it != tValues.end(); it++)
			tList.push_back(*it);

		// sample stuff from the list
		while(sValues.size() < config.maxSplits)
		{
			sValues.insert(tList[rand() % tValues.size()]);
		}
	}
	else
		sValues = tValues;

	// std::cout << sValues.size() << " Haha " << config.maxSplits << std::endl;

	// std::cout << time(NULL) << " Split Creation " << std::endl;

	std::sort(values.begin(), values.end());

	// move across the values generating the various splits
	double curValue = values[0].first;
	double leftValue, rightValue, nodeValue;

	if (config.probability == config.ConditionalProbability)
		funcptr = &DecisionTree::GetConditionalProbabilty;
		//nodeEntropy = this->ConditionalProbabilty(0, values.size(), values);
	else if (config.probability == config.AverageProbability)
		funcptr = &DecisionTree::GetAverageProbabilty;
		//nodeEntropy = this->AverageProbability(0, values.size(), values);

	switch(config.criteria)
	{
	case config.Entropy:
		nodeValue = this->Entropy(0, values.size(), values, funcptr);
		break;
	case config.GiniIndex:
		nodeValue = this->Gini(0, values.size(), values, funcptr);
		break;
	case config.Twoing:
		nodeValue = 0.0;
		break;
	case config.Orthogonal:
		nodeValue = 0.0;
		break;
	}

	double bestEval = -DBL_MAX;
	double bestSplit = DBL_MIN;
	unsigned int bestNumber = 0;
	int prev=0;

	for (unsigned int i = 1; i < values.size(); i++)
	{
		if (values[i].first == curValue)
			continue;

		if (sValues.count(curValue) == 0)
		{
			curValue = values[i].first;
			continue;
		}
/*
	// collect the data for the attributes

	vector<pair<double, InstanceID>> values (n.right + 1 - n.left);

	for( unsigned int i = n.left, j = 0; i <= n.right; i++, j++ )
	{
		values[j] = pair<double, InstanceID> (f->Evaluate(n.instances[i]), n.instances[i]);
	}

	std::sort(values.begin(), values.end());

	// move across the values generating the various splits
	double curValue = values[0].first;
	double leftEntropy, rightEntropy, nodeEntropy;

	if (config.probability == config.ConditionalProbability)
		nodeEntropy = this->ConditionalProbabilty(0, values.size(), values);
	else if (config.probability == config.AverageProbability)
		nodeEntropy = this->AverageProbability(0, values.size(), values);

	double bestEval = -DBL_MAX;
	double bestSplit = DBL_MIN;
	unsigned int bestNumber = 0;

	for (unsigned int i = 1; i < values.size(); i++)
	{
		if (values[i].first == curValue)
			continue;
*/
		// std::cout << "entered with " << curValue << std::endl;
		double split = (curValue + values[i].first) / 2.0;

		double eval, lambda;
		if (config.eLambda)
			lambda = std::pow(config.lambda, static_cast<double>(n.level));
		else
			lambda = config.lambda;

		switch(config.criteria)
		{
			case config.Entropy:
				leftValue	= this->Entropy(0, i, values, funcptr);
				rightValue	= this->Entropy(i, values.size(), values, funcptr);
				std::cout << lambda << std::endl;
				eval		= (nodeValue 
					- (i / totalInstances) * leftValue
					- ((totalInstances - i ) / totalInstances) * rightValue) 
					* (1.0 - lambda) - abs(values.size() - 2.0 * i) * lambda;
				break;
			
			case config.GiniIndex:
				leftValue	= this->Gini(0, i, values, funcptr);
				rightValue	= this->Gini(i, values.size(), values, funcptr);
				eval		= (nodeValue 
					- (i / totalInstances) * leftValue
					- ((totalInstances - i ) / totalInstances) * rightValue) 
					* (1.0 - lambda) - abs(values.size() - 2.0 * i) * lambda;
				// std::cout << leftValue << "\t" << rightValue << "\t" << nodeValue << "\t" << lambda << "\t" << eval << "\n" << std::endl;
				break;

			case config.Twoing:
				eval = this->Twoing(0,values.size(), i, values, funcptr);
				break;
			
			case config.Orthogonal:
				eval = this->Orthogonal(0,values.size(), i, values, funcptr);
				break;
	}


		/*
		if (config.probability == config.ConditionalProbability)
		{
			leftValue  = this->ConditionalProbabilty(0, i, values);
			rightValue = this->ConditionalProbabilty(i, values.size(), values);
		}
		else if (config.probability == config.AverageProbability)
		{
			this->AddProbability(&leftEntropy,&rightEntropy, prev+1, i, values);
			prev=i;
		}
		*/

		/*

		eval = (nodeEntropy - (i / totalInstances) * leftEntropy
				- ((totalInstances - i ) / totalInstances) * rightEntropy) 
				* (1.0 - lambda) - abs(values.size() - 2.0 * i) * lambda;
		*/

		if (eval > bestEval)
		{
			bestEval = eval;
			bestSplit = split;
			bestNumber = i;
		}
		curValue = values[i].first;
	}
	// std::cout << time(NULL) << " Split Over " << std::endl;
	return CandidateSplit (bestEval, bestNumber, f, bestSplit);
}

double DecisionTree::Orthogonal(unsigned int begin, unsigned int end, unsigned int middle,
	vector<pair<double, InstanceID>> &values,
	map<unsigned int, double> (DecisionTree::*funcptr) (unsigned int begin, unsigned int end,
	vector<pair<double, InstanceID>> &values))
{
	map <LabelID, double> labelLeft		= (this->*funcptr)(begin, middle, values);
	map <LabelID, double> labelRight	= (this->*funcptr)(middle, end, values);

	double partial	= 0.0;
	double left		= 0.0;
	double right	= 0.0;

	for (map<unsigned int, double>::iterator it = labelLeft.begin();
			it != labelLeft.end(); it ++)
	{
		double p1 = (*it).second;
		double p2 = 0.0;
		if(labelRight.count((*it).first) > 0)
			partial += labelRight[(*it).first] * (*it).second;
		left += (*it).second * (*it).second;
	}

	for (map<unsigned int, double>::iterator it = labelRight.begin();
			it != labelRight.end(); it ++)
		right += (*it).second * (*it).second;

	return (left > 0.0 && right > 0.0 ? partial / (sqrt(left)*sqrt(right)) : 0.0);
}


double DecisionTree::Twoing(unsigned int begin, unsigned int end, unsigned int middle,
	vector<pair<double, InstanceID>> &values,
	map<unsigned int, double> (DecisionTree::*funcptr) (unsigned int begin, unsigned int end,
	vector<pair<double, InstanceID>> &values))
{
	map <LabelID, double> labelLeft		= (this->*funcptr)(begin, middle, values);
	map <LabelID, double> labelRight	= (this->*funcptr)(middle, end, values);

	double twoing = 0.25
		* ((double)(middle - begin)/(end - begin))
		* ((double)(end - middle)/(end - begin));

	double partial = 0.0;

	for (map<unsigned int, double>::iterator it = labelLeft.begin();
			it != labelLeft.end(); it ++)
	{
		double p1 = (*it).second;
		double p2 = 0.0;
		if(labelRight.count((*it).first) > 0)
			p2 = labelRight[(*it).first];
		partial += abs(p1 - p2);
	}

	for (map<unsigned int, double>::iterator it = labelRight.begin();
			it != labelRight.end(); it ++)
	{
		if(labelRight.count((*it).first) == 0)
			partial += (*it).second;
	}
	twoing += partial * partial;

	return twoing;
}


void DecisionTree::AddProbability(double *v,double* u,unsigned int begin, unsigned int end, vector<pair<double, InstanceID>> &values){
	double total = 0.0;
	map <LabelID, double> labels;
	// calculate the left entropy
	for (unsigned int j = begin; j < end; j++)
	{
		const map<LabelID, double> &lbls = this->instanceData.Labels(values[j].second);
		for (map<LabelID, double>::const_iterator it = lbls.cbegin(); it != lbls.cend(); it++)
		{
			labels[(*it).first] += (*it).second;
			total += (*it).second;
		}
	}
	double p = 0.0;
	for (map<unsigned int, double>::iterator it = labels.begin();
			it != labels.end(); it ++)
	{
		p = (*it).second / total;
		*v += ( p > 0.0 ? -p * log(p) : 0.0);
		*u -= ( p > 0.0 ? -p * log(p) : 0.0);
	}

}

double DecisionTree::AverageProbability(unsigned int begin, unsigned int end, vector<pair<double, InstanceID>> &values)
{
	double total = 0.0;
	map <LabelID, double> labels;
	// calculate the left entropy
	for (unsigned int j = begin; j < end; j++)
	{
		const map<LabelID, double> &lbls = this->instanceData.Labels(values[j].second);
		for (map<LabelID, double>::const_iterator it = lbls.cbegin(); it != lbls.cend(); it++)
		{
			labels[(*it).first] += (*it).second;
			total += (*it).second;
		}
	}

	double entropy = 0.0;
	double p = 0.0;

	for (map<unsigned int, double>::iterator it = labels.begin();
			it != labels.end(); it ++)
	{
		p = (*it).second / total;
		entropy += ( p > 0.0 ? -p * log(p) : 0.0);
	}

	return entropy;
}

double DecisionTree::Gini(unsigned int begin, unsigned int end,
	vector<pair<double, InstanceID>> &values,
	map<unsigned int, double> (DecisionTree::*funcptr) (unsigned int begin, unsigned int end,
	vector<pair<double, InstanceID>> &values))
{
	map <LabelID, double> labels = (this->*funcptr)(begin, end, values);
	double gini = 0.0;

	for (map<unsigned int, double>::iterator it = labels.begin();
			it != labels.end(); it ++)
	{
		double p = (*it).second;
		gini -= ( p > 0.0 ? p * p : 0.0);
	}
	gini += 1.0;
	return gini;
}

double DecisionTree::Entropy(unsigned int begin, unsigned int end,
	vector<pair<double, InstanceID>> &values,
	map<unsigned int, double> (DecisionTree::*funcptr) (unsigned int begin, unsigned int end,
	vector<pair<double, InstanceID>> &values))
{
	map <LabelID, double> labels = (this->*funcptr)(begin, end, values);
	double entropy = 0.0;

	for (map<unsigned int, double>::iterator it = labels.begin();
		it != labels.end(); it ++)
		entropy += ( (*it).second > 0.0 ? -(*it).second * log((*it).second) : 0.0);
	return entropy;
}

double DecisionTree::ConditionalProbabilty(unsigned int begin, unsigned int end, vector<pair<double, InstanceID>> &values)
{
	map <unsigned int, double> labels;
	for (unsigned int j = begin; j < end; j++)
	{
		const map<LabelID, double> &lbls = this->instanceData.Labels(values[j].second);
		for (map<LabelID, double>::const_iterator it = lbls.cbegin(); it != lbls.cend(); it++)
			labels[(*it).first] += (*it).second / (lbls.size() * (end - begin));
	}

	double entropy = 0.0;

	for (map<unsigned int, double>::iterator it = labels.begin();
			it != labels.end(); it ++)
		entropy += ( (*it).second > 0.0 ? -(*it).second * log((*it).second) : 0.0);
	return entropy;

}

map<unsigned int, double> DecisionTree::GetAverageProbabilty(unsigned int begin, unsigned int end, vector<pair<double, InstanceID>> &values)
{
	double total = 0.0;
	map <LabelID, double> labels;
	// calculate the left entropy
	for (unsigned int j = begin; j < end; j++)
	{
		const map<LabelID, double> &lbls = this->instanceData.Labels(values[j].second);
		for (map<LabelID, double>::const_iterator it = lbls.cbegin(); it != lbls.cend(); it++)
		{
			labels[(*it).first] += (*it).second;
			total += (*it).second;
		}

	}

	for (map<unsigned int, double>::iterator it = labels.begin();
		it != labels.end(); it ++)
		(*it).second /= total;

	return labels;
}

map<unsigned int, double> DecisionTree::GetConditionalProbabilty(unsigned int begin, unsigned int end, vector<pair<double, InstanceID>> &values)
{
	double total = 0.0;
	map <LabelID, double> labels;

	for (unsigned int j = begin; j < end; j++)
	{
		const map<LabelID, double> &lbls = this->instanceData.Labels(values[j].second);
		for (map<LabelID, double>::const_iterator it = lbls.cbegin(); it != lbls.cend(); it++)
			labels[(*it).first] += (*it).second / (lbls.size() * (end - begin));
	}

	return labels;
}


vector <SplitFeature *> *DecisionTree::CandidateSplitFeature( const Node &n )
{
	if (this->config.features == this->config.SingleFeature)
	{
		set<FeatureID> considered;
		set<FeatureID> candidates;

		// simply generate the multiple features
		while (considered.size() < this->instanceData.NumFeatures()
			&& candidates.size() < this->config.numFeatures)
		{
			FeatureID f = this->GetRandomFeature();
			if (considered.count(f) == 1)
				continue;
			
			considered.insert(f);
			// variance threshold
			SingleFeature *g=new SingleFeature(this->instanceData,f);
			if (Variance(g, n) <= this->config.varianceThreshold ){
				delete g;
				continue;
			}
			delete g;
			candidates.insert(f);
		}

		// handle the return value here
		vector <SplitFeature *> *t = new vector<SplitFeature *>(candidates.size());
		unsigned int i = 0;
		for(set<FeatureID>::iterator it = candidates.begin();
				it != candidates.end(); it++, i++)
			t->operator[](i) = new SingleFeature(this->instanceData, *it);
		//std::cout << "Considered : " << considered.size() << std::endl;
		//std::cout << "Candidates : " << candidates.size() << std::endl;
		return t;
	}
	else{
		int param = 5;
		vector <SplitFeature *> *t=new vector<SplitFeature *>(param*param);

		for(int i=0;i<param;i++){

			set<FeatureID> considered;

			vector<FeatureID> candidates;

			while(candidates.size()<NUM_MULTI_FEATURES){

				FeatureID f = this->GetRandomFeature();

				if (considered.count(f) == 1)

					continue;

				considered.insert(f);

				candidates.push_back(f);

			}
			int l=0;
			for(int j=0;j<param;j++)
			{
				int weight=0;
				for(int k=0;k<NUM_MULTI_FEATURES;k++)
				{
					int l=rand()%2;
					weight|=(l<<k);
				}
				MultipleFeature *mf=new MultipleFeature(this->instanceData,weight);
				
				for(int k=0;k<NUM_MULTI_FEATURES;k++)
				{
					mf->insertFeature(candidates[k]);
				}
				l++;
				/*if (Variance(mf, n) <= this->config.varianceThreshold && l<100){
					delete mf;
					j--;
					continue;
				}*/

				t->operator[](i*param+j)=mf;

			}
		}
		return t;
	}
	return 0;
}

FeatureID DecisionTree::GetRandomFeature( void )
{
	// return this->instanceData.NumToFeatureID(rand() % this->instanceData.NumFeatures());
	return this->instanceData.NumToFeatureID(this->featureSampler(this->gen) % this->instanceData.NumFeatures());
}



double DecisionTree::Variance(SplitFeature *f, const Node &n)
{
	double x = 0.0;
	double xSquare = 0.0;
	for(unsigned int i = n.left; i <= n.right; i++)
	{
		double t = f->Evaluate(n.instances[i]);
		x += t;
		xSquare += t * t;
	}

	xSquare /= (n.right - n.left + 1);
	x       /= (n.right - n.left + 1);
	x       *= x;

	return xSquare - x;
}

template <class T>
void print_list(std::ostream &o, vector <T> list, char*s)
{
	unsigned int total = list.size();
	unsigned int cur   = 0;
	o << s << "\t";
	for (vector <T>::iterator it = list.begin(); it != list.end(); it++, cur++)
	{
		o << *it;
		if (cur != total - 1)
			o << "\t";
		else
			o << std::endl;
	}
}

void DecisionTree::DumpTree(std::ostream &o)
{
	o << "Tree_" << this->id << std::endl;
	o << "Time\t" << this->start << "\t" << this->end << std::endl;
	o << "Instances\t" << this->leftChild.size() << std::endl;

	// dump

	unsigned int total = this->_splitFeatures.size();
	unsigned int cur   = 0;
	o << "SplitFeatures" << "\t";
	for (vector <SplitFeature *>::iterator it = this->_splitFeatures.begin();
		it != this->_splitFeatures.end(); it++, cur++)
	{
		if (*it == NULL)
			o << "-1";
		else
		{
			if(this->config.features == this->config.SingleFeature)
				o << *it;
			else
			{
				MultipleFeature* f = (MultipleFeature*)(*it);
				o << "M\t";
				for (int i = 0; i<20; i++)
				{
					o << f->features[i] << "\t";
				}
				o << f->weight;
			}
		}
		if (cur != total - 1)
			o << "\t";
		else
			o << std::endl;
	}	
	
	print_list(o, _splitThresholds, "SplitThresholds");
	print_list(o, leftChild,        "LeftChild");
	print_list(o, rightChild,       "RightChild");

	// std::cout << labelDist.size() << " this is the labels " << std::endl;
	for (vector <map<LabelID, double>>::iterator it = labelDist.begin();
		it != labelDist.end(); it++)
	{
		cur = 0;
		total = (*it).size();
		for(map<LabelID, double>::iterator it1 = (*it).begin();
			it1 != (*it).end(); it1++, cur++)
		{
			o << (*it1).first << ":" << (*it1).second;
			if (cur != total -1)
				o << "\t";
			else
				o << std::endl;
		}
	}

}
