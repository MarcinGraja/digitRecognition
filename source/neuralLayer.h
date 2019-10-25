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
	size_t size();
	std::vector<double> getValues();
	void setValues(std::vector<double> values, bool acceptDifferentSizes);
	void updateValues(std::vector<double> prevValues);
};

