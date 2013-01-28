#include "..\RandomForest\InstanceData.h"

#include <fstream>
#include <iostream>
#include <map>
#include <set>

using std::ifstream;
using std::map;
using std::set;

void InstanceData::FeatureLoader ( string filename )
{
	ifstream inputFile(filename);
	if (!inputFile)
	{
		std::cout << "Invalid feature file" << std::endl;
		exit(-1);
	}
	string inputLine;
	
	// get data stats
	std::getline(inputFile, inputLine);
	
	string::size_type pos = inputLine.find(' ');
	this->_numInstances = atoi(inputLine.substr(0, pos).c_str());
	this->_numFeatures  = atoi(inputLine.substr(pos).c_str());

	this->_features = new map<FeatureID, double>[this->_numInstances];
	
	// consume the remaining data
	unsigned int row = 0;
	while(getline(inputFile, inputLine))
	{
		// split the line to get the features
		// inputLine.erase(remove_if(inputLine.begin(), inputLine.end(), isspace), inputLine.end());

		string::size_type lastPos = -1;
		unsigned int attr;
		double value;
		int fvSep;
		
		while ((pos = inputLine.find(' ', lastPos + 1)) != string::npos)
		{
			fvSep = inputLine.find(':', lastPos + 1);
			
			attr = atoi(inputLine.substr(lastPos + 1, fvSep).c_str());
			value = static_cast<double>(atof(inputLine.substr(fvSep + 1, pos).c_str()));
			this->_addFeature(row, attr, value);
			
			lastPos = pos;
		}
		
		// handle the last case
		fvSep = inputLine.find(':', lastPos + 1);
		attr = atoi(inputLine.substr(lastPos + 1, fvSep).c_str());
		value = static_cast<double>(atof(inputLine.substr(fvSep + 1).c_str()));
		this->_addFeature(row, attr, value);

		row ++;
	}
}

void InstanceData::LabelLoader   ( string filename )
{
	ifstream inputFile(filename);
	string inputLine;

	if (!inputFile)
	{
		std::cout << "Invalid label file : " << filename << std::endl;
		exit(-1);
	}	// get data stats
	std::getline(inputFile, inputLine);
	
	string::size_type pos = inputLine.find(' ');

	this->_numLabels = atoi(inputLine.substr(pos).c_str());
	this->_labels    = new map<LabelID, double>[this->_numInstances];

	// consume the remaining data
	unsigned int row = 0;
	while(getline(inputFile, inputLine))
	{
		// split the line to get the features
		// inputLine.erase(remove_if(inputLine.begin(), inputLine.end(), isspace), inputLine.end());
			
		string::size_type lastPos = -1;
		unsigned int attr;
		int fvSep;
		double value;
		while ((pos = inputLine.find(' ', lastPos + 1)) != string::npos)
		{
			fvSep = inputLine.find(':', lastPos + 1);
			attr = atoi(inputLine.substr(lastPos + 1, fvSep).c_str());
			value = static_cast<double>(atof(inputLine.substr(fvSep + 1, pos).c_str()));
			this->_addLabel(row, attr, value);

			lastPos = pos;
		}
		
		// handle the last case
		fvSep = inputLine.find(':', lastPos + 1);
		attr = atoi(inputLine.substr(lastPos + 1, fvSep).c_str());
		value = static_cast<double>(atof(inputLine.substr(fvSep + 1, pos).c_str()));
		this->_addLabel(row, attr,value);

		row ++;
	}
}
