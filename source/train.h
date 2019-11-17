#pragma once
#include <vector>
#include "NeuralNetwork.h"
class train
{
	std::vector <int> dimensions;
	int trainingSetSize;
	double *trainingData;
	int *trainingLabels;
	NeuralNetwork network;
public:
	train(std::vector <int> dimensions);
	~train();
	double * fetchData(std::ifstream & data, int itemSize, int itemCount);
	int * fetchLabels(std::ifstream & labels, int itemCount);
	double printHitrateInRange(int start, int end);
	void start(int runs);
};