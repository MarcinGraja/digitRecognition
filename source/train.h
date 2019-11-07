#pragma once
#include <vector>
class train
{
	std::vector <int> dimensions;
public:
	train(std::vector <int> dimensions);
	~train();
	void start(int runs);
};