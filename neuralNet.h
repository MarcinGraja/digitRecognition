#pragma once
#include <vector>
class neuralNet
{
	std::vector<neuralLayer> layers;
public:
	neuralNet(std::vector<int> dimensions);
	std::vector<double> run(std::vector<double> data);
	std::vector<double> run(unsigned char * data);
	void initWeights();
	~neuralNet();
};

