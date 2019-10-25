#pragma once
#include "cell.h"
#include <vector>
class neuralLayer
{
	std::vector<cell> layer;
public:
	neuralLayer(int cellCount);
	void setWeights(std::vector<std::vector<double>> weights);
	void initWeights(int count);
	~neuralLayer();
	int size();
	std::vector<double> getValues();
	void setValues(std::vector<double> values, bool acceptDifferentSizes = false);
	void updateValues(std::vector<double> prevValues);
};
