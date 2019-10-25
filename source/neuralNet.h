#pragma once
#include <vector>
#include "neuralLayer.h"
class neuralNet
{
	std::vector<neuralLayer> layers;
public:
	neuralNet(std::vector<int> dimensions);
	std::vector<double> run(std::vector<double> data);
	void initWeights();
	~neuralNet();
};

