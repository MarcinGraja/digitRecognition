#pragma once
#include "cell.h"
#include <vector>
class neuralLayer
{
	neuralLayer *prevLayer;
	std::vector<cell> layer;
public:
	neuralLayer(int cellCount, neuralLayer *prevLayer);
	void setWeights(std::vector<std::vector<double>> weights);
	~neuralLayer();
	int size();
	std::vector<double> getValues();
	double getValue(int i);
	void setValues(std::vector<double> values);
	void updateValues(std::vector<double> prevValues);
	neuralLayer * getPreviousLayer();
	void backPropagate(neuralLayer * youDumbFuck, neuralLayer * outputLayer, double sumOutput, std::vector<double> error, double sumError, bool isOutput);
};

