#pragma once
class cell
{
public:
	void setValue(double value);
	void updateValue(cell *previousColumn, int count);
	void setMultipliers(double *value, int count);
	cell();
	~cell();
};

