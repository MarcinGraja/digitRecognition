#pragma once
#include <vector>
class cell
{
	double value;
	std::vector <double> weights;
public:
	void initWeights(int count);
	void setValue(double value);
	void updateValue(std::vector<double> previousColumn);
	void setWeights(std::vector<double> weights);
	std::vector<double> getWeights();
	void updateWeights(double maxPercentageChange, int count);
	double getValue();
	cell();
	~cell();
};

