#pragma once
#include <vector>
class neuralLayer;
class cell
{
	double value;
	std::vector <double> weights;
	std::vector <double> newWeights;
	void initWeights(int count);
public:
	void setValue(double value);
	void updateValue(std::vector<double> previousColumn);
	void setWeights(std::vector<double> weights);
	std::vector<double> getWeights();
	double getValue();
	void backPropagate(neuralLayer * prevLayer, neuralLayer * outputLayer, double sumOutput, std::vector<double> error, double sumError, bool isOutput, int index);
	void updateWeights();
	cell(int prevCount);
	cell();
	~cell();
};