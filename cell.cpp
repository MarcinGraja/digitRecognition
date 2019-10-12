#include "cell.h"
#include <random>
#include <ctime>

class cell
{
	double value = 0;
	double *multipliers = NULL;
public:
	void initMultipliers(int count)
	{
		if (count == 0) return;
		multipliers = new double[count];
		for (int i = 0; i < count; i++)
		{

			multipliers[i] = (rand() % 2000 - 1000) / 1000.0;
		}
	}
	void setValue(double value)
	{
		this->value = value;
	}
	void updateValue(cell *previousColumn, int count)
	{
		value = 0;
		for (int i = 0; i < count; i++)
		{
			value += previousColumn[i].getValue() * multipliers[i];
		}
	}
	void setMultipliers(double *value, int count)
	{
		if (count <= 0) throw std::invalid_argument("prevcount should be >0");
		for (int i = 0; i < count; i++)
		{
			multipliers[i] = value[i];
		}
	}
	double *getMultipliers()
	{
		return multipliers;
	}
	void updateMultipliers(double maxPercentageChange, int count)
	{
		for (int i = 0; i < count; i++)
		{
			double absoluteNewValue = rand() % 1000 * maxPercentageChange / 1000.0 / 100 * multipliers[i] + 0.001;
			multipliers[i] = rand() % 2 == 0 ? absoluteNewValue : -absoluteNewValue;
		}
	}
	double getValue()
	{
		return value;
	}
};
cell::cell()
{
}


cell::~cell()
{
}
