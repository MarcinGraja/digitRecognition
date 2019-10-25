#include "cell.h"
#include <random>
#include <ctime>
#include <vector>

void cell::initWeights(int count)
{
	if (count == 0) return;
	weights.reserve(count);
	for (int i = 0; i < count; i++)
	{
		weights[i] = (rand() % 2000 - 1000) / 1000.0;
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
		value += previousColumn.at(i) * weights[i];
	}
}
void cell::setWeights(std::vector<double> weights)
{
	this->weights = weights;
}
std::vector <double> cell::getWeights()
{
	return weights;
}
void cell::updateWeights(double maxPercentageChange, int count)
{
	for (int i = 0; i < count; i++)
	{
		double absoluteNewValue = rand() % 1000 * maxPercentageChange / 1000.0 / 100 * weights[i] + 0.001;
		weights[i] = rand() % 2 == 0 ? absoluteNewValue : -absoluteNewValue;
	}
}
double cell::getValue()
{
	return value;
}
cell::cell()
{
}


cell::~cell()
{
}
