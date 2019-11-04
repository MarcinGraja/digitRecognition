#include "neuralLayer.h"
#include <vector>
#include <stdexcept>
#include <string>
neuralLayer::neuralLayer(int cellCount, neuralLayer *prevLayer)
{
	this->prevLayer = prevLayer;
	layer.reserve(cellCount);
	{
		for (int i = 0; i < cellCount; i++)
		{
			if (prevLayer != nullptr)
				layer.push_back(cell(prevLayer->size()));
			else layer.push_back(cell());
		}
	}
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
int neuralLayer::size()
{
	return (int) layer.size();
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
double neuralLayer::getValue(int i)
{
	return layer.at(i).getValue();
}
void neuralLayer::setValues(std::vector<double> values)
{
	for (int i = 0; i < layer.size() && i < values.size(); i++)
	{
		layer.at(i).setValue(values.at(i));
	}
}
void neuralLayer::updateValues(std::vector <double> prevValues)
{
	for (int i = 0; i < layer.size(); i++)
	{
		layer[i].updateValue(prevValues);
	}
}
neuralLayer *neuralLayer::getPreviousLayer()
{
	return prevLayer;
}
void neuralLayer::backPropagate(neuralLayer *prev, neuralLayer *outputLayer, double sumOutput, std::vector <double> error, double sumError, bool isOutput)
{
	for (int i = 0; i < layer.size(); i++)
	{
		layer.at(i).backPropagate(prev, outputLayer, sumOutput, error, sumError, isOutput, i);
	}
	if (!isOutput)
		for (cell c : layer)
		{
			c.updateWeights();
		}
}
