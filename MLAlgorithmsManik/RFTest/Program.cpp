#include <algorithm>
#include <sstream>
#include <vector>
#include "..\RandomForest\InstanceData.h"
#include "..\RandomForest\DecisionTree.h"

using std::ifstream;
using std::stringstream;
using std::vector;

class SimpleDT
{
public:
	unsigned int                    instances;
	vector <SplitFeature *>         _splitFeatures;
	vector <double>                 _splitThresholds;
	vector <unsigned int>           leftChild;
	vector <unsigned int>		    rightChild;
	vector <map<LabelID, double>>   labelDist;
	string                          id;

	map<LabelID, double>& GetDistribution(InstanceID inst)
	{
		unsigned int i = 0;
		double val = 0;
		while(_splitFeatures[i] != NULL)
		{
			val = _splitFeatures[i]->Evaluate(inst);
			// std::cout << "came here " << inst << " value " << val << std::endl;
			if (val < this->_splitThresholds[i])
				i = this->leftChild[i];
			else
				i = this->rightChild[i];
		}
		return this->labelDist[i];
	}
};

class SimpleRF
{
public:
	SimpleRF(string filename, InstanceData &data) : _data(data)
	{
		string inputLine;
		std::ifstream inputFile(filename);
		while(std::getline(inputFile, inputLine))
		{
			size_t found;
			// extract the first word
			stringstream abc(inputLine);
			string word;
			abc >> word;
			
			found = word.find("Tree_");
			if ((int)found == 0)
			{
				std::cout<<"New Tree"<<std::endl;
				trees.push_back(SimpleDT());
				trees[trees.size() - 1].id = word;
				std::getline(inputFile, inputLine);
				this->getInstances(inputFile, trees[trees.size() - 1]);
				this->readFeatures(inputFile, trees[trees.size() - 1]._splitFeatures, data);
				this->readVector(inputFile, trees[trees.size() - 1]._splitThresholds);
				this->readVector(inputFile, trees[trees.size() - 1].leftChild);
				this->readVector(inputFile, trees[trees.size() - 1].rightChild);

				for(unsigned int i = 0 ; i < trees[trees.size() - 1].instances; i++)
				{
					//std::cout<<i<<std::endl;
					trees[trees.size() - 1].labelDist.push_back(map<LabelID, double>());
					this->readDistribution(inputFile, trees[trees.size() - 1].labelDist[i]);
				}
			}
		}

		/*
		std::cout << trees.size() << std::endl;
		std::cout << trees[0].instances << std::endl;
		for (map<LabelID, double>::iterator it = trees[0].labelDist[1].begin();
			it != trees[0].labelDist[1].end(); it++)
			std::cout << (*it).first << ":" << (*it).second << std::endl;
		*/

	}

	void readFeatures(std::ifstream &i, vector<SplitFeature *>& list, InstanceData &data)
	{
		string line;
		std::getline(i, line);
		stringstream abc(line);

		abc >> line;
		while(abc >> line)
		{
			if (line[0] == 'S')
				list.push_back(SingleFeature::Parse(line, data));
			else if (line[0] == '-')
				list.push_back(NULL);
			else if(line[0] == 'M')
			{
				MultipleFeature *temp = new MultipleFeature(data, 0);
				for(int j = 0; j< NUM_MULTI_FEATURES; j++)
				{
					abc >> line;
					temp->insertFeature(atoi(line.c_str()));
				}
				abc >> line;
				temp->weight = atoi(line.c_str());
				list.push_back(temp);
			}
		}
	}

	void getInstances(std::ifstream &i, SimpleDT &dt)
	{
		string line;
		std::getline(i, line);

		stringstream abc(line);
		abc >> line >> dt.instances;
	}

	void readDistribution(std::ifstream &i, map<LabelID, double> &dist)
	{
		string::size_type pos;
		string inputLine;
		std::getline(i, inputLine);
		// split the line to get the features
		// inputLine.erase(remove_if(inputLine.begin(), inputLine.end(), isspace), inputLine.end());

		string::size_type lastPos = -1;
		LabelID attr;
		double value;
		int fvSep;
		
		while ((pos = inputLine.find('\t', lastPos + 1)) != string::npos)
		{
			fvSep = inputLine.find(':', lastPos + 1);
			
			attr = atoi(inputLine.substr(lastPos + 1, fvSep).c_str());
			value = static_cast<double>(atof(inputLine.substr(fvSep + 1, pos).c_str()));
			dist[attr] =  value;
			
			lastPos = pos;
		}
		
		// handle the last case
		fvSep = inputLine.find(':', lastPos + 1);
		attr = atoi(inputLine.substr(lastPos + 1, fvSep).c_str());
		value = static_cast<double>(atof(inputLine.substr(fvSep + 1).c_str()));
		dist[attr] = value;
	}


	template<class T>
	void readVector(std::ifstream &i, vector<T>& list)
	{
		string line;
		std::getline(i, line);
		stringstream abc(line);

		T val;
		abc >> line;
		while(abc >> val)
			list.push_back(val);
	}
	vector<SimpleDT> trees;
	InstanceData &_data;
};

void Evaluate(SimpleRF &rf, InstanceData &data,std::ostream &o)
{
	double p1 = 0, p5 = 0, p10 = 0;
	for(unsigned int i = 0; i < data.NumInstances(); i++)
	{
		map<LabelID, double> mDist;
		// std::cout << rf.trees.size() << std::endl;
		for (vector<SimpleDT>::iterator it = rf.trees.begin(); it != rf.trees.end(); it++)
		{
			map<LabelID, double> &c = (*it).GetDistribution(i);
			for(map<LabelID, double>::iterator it1 = c.begin(); it1 != c.end(); it1++)
			{
				if (mDist.count((*it1).first) == 0)
					mDist[(*it1).first] = (*it1).second;
				else
					mDist[(*it1).first] += (*it1).second;
			}
		}

		vector<pair<double, LabelID>> labels;
		for(map<LabelID, double>::iterator it1 = mDist.begin(); it1 != mDist.end(); it1++)
		{
			labels.push_back(pair<double, LabelID>(-(*it1).second, (*it1).first));
		}
		sort(labels.begin(), labels.end());

		for(unsigned int i1 = 0; i1 < labels.size() && i1 < 10; i1++)
		{
			// std::cout << labels[i1].second << std::endl;
			unsigned int count = data.Labels(i).count(labels[i1].second);
			if (i1 < 1)
				p1 += count;
			if (i1 < 5)
				p5 += count;
			if (i1 < 10)
				p10 += count;
			// std::cout << count << "\t" << p1 << "\t" << p5 << "\t" << p10 << std::endl;
		}
		// std::cout << p1 << "\t" << p5 << "\t" << p10 << std::endl;
	}

	o
		<< p1 / data.NumInstances() << "\t"
		<< p5 / (5.0 * data.NumInstances()) << "\t"
		<< p10 / (10.0 * data.NumInstances()) << std::endl;
	int x=0;
}

int main(int argc, char*argv[])
{
	//for(int i=0;i<10;i++){
		string featureFile	("D:\\Studies\\Sem7\\BTP\\RFData\\CSVZips\\Csv-Format\\BIBTEX\\tstFtMat.0.csv");
		string labelFile	("D:\\Studies\\Sem7\\BTP\\RFData\\CSVZips\\Csv-Format\\BIBTEX\\tstLblMat.0.csv");
		std::cout << labelFile << std::endl;

		InstanceData data(featureFile, labelFile);
		string p("D:\\Studies\\Sem7\\BTP\\RFData\\CSVZips\\Csv-Format\\BIBTEX\\result.txt");
		//sprintf(num,"%d.txt",i);
		SimpleRF simpleRF(p, data);
		std::cout << time(NULL) << std::endl;
		Evaluate(simpleRF, data, std::cout);
		std::cout << time(NULL) << std::endl;
	//}
	return 0;
}