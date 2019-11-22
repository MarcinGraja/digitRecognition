#pragma once
#include <vector>
#include <fstream>
#include "NeuralNetwork.h"
class train
{
	std::vector <int> dimensions;
	int trainingSetSize;
	int testingSetSize;
	double *trainingData;
	int *trainingLabels;
	double *testingData;
	int *testingLabels;
	std::ofstream log;
	std::ofstream csvLog;
public:
	train(std::vector<int> dimensions, int trainingSetSize, int testingSetSize);
	~train();
	double *printHitrateInRange(int start, int end, NeuralNetwork network, std::string m, std::string csvM);
	void run(int runs);
	void findHyperParameters(int runs);
	void backpropagate(NeuralNetwork & network, double base, double step, int runs, double & returned);
	void probTrashChoosesOptimalLearningRate(double runs);
};