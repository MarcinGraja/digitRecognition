#include "neuralLayer.h"
#include <vector>
#include <stdexcept>
#include <string>
neuralLayer::neuralLayer(int cellCount)
{
	layer.reserve(cellCount);
}
neuralLayer::~neuralLayer(){}
void neuralLayer::setWeights(std::vector<std::vector<double>> weights)
{
	if (layer.size() != weights.size())
	{
		throw std::invalid_argument("Invalid argument: layer size(" + layer.size() + std::string(") doesn't number of weights (") + std::to_string((int)(weights.size())) + ")");
	}
	for (int i = 0; i < layer.size(); i++)
	{
		layer.at(i).setWeights(weights.at(i));
	}
}
 void neuralLayer::initWeights(int count)
{
	for (cell c : layer)
	{
		c.initWeights(count);
	}
}
size_t neuralLayer::size()
{
	return layer.size();
}
std::vector <double> neuralLayer::getValues()
{
	std::vector <double> values;
	values.reserve(layer.size());
	for (cell c : layer)
	{
		values.push_back(c.getValue());
	}
	return values;
}
void neuralLayer::setValues(std::vector<double> values, bool acceptDifferentSizes = false)
{
	if (!acceptDifferentSizes && layer.size() != values.size())
	{
		std::string s = "invalid argument; layer size(" + std::to_string(layer.size()) + ") doesn't equal values size(" + std::to_string(values.size()) + ")" ;
		throw std::invalid_argument(s);
	}
	for (int i = 0; i < layer.size() && i < values.size(); i++)
	{
		layer.at(i).setValue(values.at(i));
	}
}
void neuralLayer::updateValues(std::vector <double> prevValues)
{
	for (cell c : layer)
	{
		c.updateValue(prevValues);
	}
}
