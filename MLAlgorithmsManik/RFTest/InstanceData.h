#ifndef __INSTANCEDATA_H__
#define __INSTANCEDATA_H__

#include <cassert>
#include <map>
#include <set>
#include <string>

using std::map;
using std::set;
using std::string;

typedef unsigned int FeatureID;
typedef unsigned int InstanceID;
typedef unsigned int LabelID;

class InstanceData
{
public:
	InstanceData (string featureFile, string labelFile)
	{
		this->FeatureLoader(featureFile);
		this->LabelLoader(labelFile);
	}
	inline unsigned int NumInstances ( void ) const
	{ 
		return _numInstances;
	}

	inline unsigned int NumLabels    ( void ) const
	{ 
		return _numLabels;
	}

	inline unsigned int NumFeatures  ( void ) const
	{
		return _numFeatures;
	}

	const set<LabelID> &Labels( InstanceID instance) const
	{
		return _labels[instance];
	}
	
	inline double GetFeature(InstanceID i, FeatureID f) const
	{
		assert(i >= 0 && i < _numInstances && f >= 0 && f < _numFeatures);
		return _features[i].count(f) == 0 ? 0.0 : _features[i][f];
	}

	inline FeatureID NumToFeatureID (unsigned int i) const
	{
		return i;
	}

private:
	void FeatureLoader( string filename );
	void LabelLoader  ( string filename );
	inline void _addFeature  ( InstanceID i, FeatureID f, double v)
	{
		assert(i >= 0 && i < _numInstances && f >= 0 && f < _numFeatures);
		_features[i][f] = v;
	}
	inline void _addLabel ( InstanceID i, LabelID l)
	{
		assert(i >= 0 && i < _numInstances && l >= 0 && l < _numLabels);
		_labels[i].insert(l);
	}

	map<FeatureID, double> *_features;
	set<LabelID>           *_labels;
	unsigned int            _numFeatures;
	unsigned int            _numInstances;
	unsigned int            _numLabels;

};
#endif