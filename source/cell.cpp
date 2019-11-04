#include "cell.h"
#include <random>
#include <ctime>
#include <vector>
#include "neuralLayer.h"
void cell::initWeights(int count)
{
	if (count == 0) return;
	weights.reserve(count);
	for (int i = 0; i < count; i++)
	{
		weights.push_back((rand() % 2000 - 1000) / 1000.0);
	}
}
void cell::setValue(double value)
{
	this->value = value;
}
void cell::updateValue(std::vector <double> previousColumn)
{
	value = 0;
	for (int i = 0; i < weights.size(); i++)
	{
		value += previousColumn.at(i) * weights.at(i);
		
	}
	value = 1.0 / (1 + std::exp(-value));
}
void cell::setWeights(std::vector<double> weights)
{
	this->weights = weights;
}
std::vector <double> cell::getWeights()
{
	return weights;
}
double cell::getValue()
{
	return value;
}

void cell::backPropagate(neuralLayer * prevLayer,  neuralLayer * outputLayer, double sumOutput, std::vector <double> error, double sumError, bool isOutput, int index)

{
	double alpha = 0.5;
	if (isOutput)
		for (int i = 0; i < weights.size(); i++)
		{
			weights.at(i) -= alpha * error.at(index) * prevLayer->getValue(i);
		}
	else
	{
		if (newWeights.size() == 0)
		{
			newWeights.resize(weights.size(), 0);
		}
		for (int i = 0; i < weights.size(); i++)
		{
			newWeights.at(i) -= alpha * sumError * sumOutput * weights.at(i) * (1 - getValue());
		}
	}
}
void cell::updateWeights()
{
	weights.swap(newWeights);
	newWeights.clear();
}
cell::cell(int prevCount)
{
	initWeights(prevCount);
}
cell::cell() 
{
}

cell::~cell(){}
