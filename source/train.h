#pragma once
#include <vector>
class train
{
	std::vector <int> dimensions;
	int trainingSetSize;
	double *data;
	int *labels;
	NeuralNetwork network;
public:
	train(std::vector <int> dimensions);
	~train();
	double * fetchData(std::ifstream & data, int itemSize, int itemCount);
	int * fetchLabels(std::ifstream & labels, int itemCount);
	void printHitrateInRange(int start, int end);
	void start(int runs);
};