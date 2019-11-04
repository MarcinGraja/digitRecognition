#include <fstream>
#include <vector>
#include <stdexcept>
#include "neuralNet.h"
neuralNet::neuralNet(std::vector<int> dimensions)
{
	if (dimensions.size() < 2)
	{
		throw std::invalid_argument("Invalid argument; vector specifying dimensions has to have at least size 2. Current size: " + dimensions.size() + '\n');
	}
	for (int i = 0; i < dimensions.size(); i++)
	{
		layers.push_back(neuralLayer(dimensions.at(i), i > 0? &layers[i-1] : nullptr));
	}
}
std::vector <double>neuralNet::run(std::vector <double> data)
{
	layers.at(0).setValues(data);
	for (int i = 1; i < layers.size(); i++)
	{
		layers.at(i).updateValues(layers.at(i - 1).getValues());

	}
	return layers.at(layers.size()-1).getValues();
}
void neuralNet::backPropagate(std::vector <double> error)
{
	double sumError = 0;
	double sumOutput = 0;
	for (double d : error)
	{
		sumError += d;
	}
	for (double d : layers.back().getValues())
	{
		sumOutput += d;
	}
	for (int i = 1; i < layers.size(); i++)
	{
		layers.at(i).backPropagate(&layers.at(i-1), &layers.back(), sumError, error, sumError, i == layers.size() - 1);
	}
}

neuralNet::~neuralNet()
{
}