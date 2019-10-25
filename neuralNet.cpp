#include <fstream>
#include "neuralNet.h"
#include "neuralLayer.h"
#include <vector>
#include <stdexcept>

neuralNet::neuralNet(std::vector<int> dimensions)
{
	if (dimensions.size() < 2)
	{
		throw std::invalid_argument("Invalid argument; vector specifying dimensions has to have at least size 2. Current size: " + dimensions.size() + '\n');
	}
	for (int i = 0; i < dimensions.size(); i++)
	{
		layers.push_back(neuralLayer(dimensions.at(i)));
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
void neuralNet::initWeights()
{
	for (int i = 1; i < layers.size(); i++)
	{
		for (int j = 0; j < layers.at(j).size(); j++)
		{
			layers.at(i).initWeights(layers.at(i-1).size());
		}
	}
}
neuralNet::~neuralNet()
{
}
