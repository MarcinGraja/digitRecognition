#pragma once
#include <ctime>
#include <random>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "NeuralNetwork.h"
#include "train.h"
#include "Eigen/core"
#include <chrono>
#define NOTIFY_FREQUENTLY false
typedef std::chrono::high_resolution_clock Clock;
train::train(std::vector<int> dimensions) : network(NeuralNetwork(dimensions))
{
	this->dimensions = dimensions;
	trainingSetSize = 60000;
	std::ifstream dataFstream("data/train-images.idx3-ubyte", std::fstream::binary);
	std::ifstream labelsFstream("data/train-labels.idx1-ubyte", std::fstream::binary);
	if (!dataFstream.is_open() || !labelsFstream.is_open())
	{
		std::cerr << "data or labels not loaded; Data open:  " << dataFstream.is_open() << "labels open: " << labelsFstream.is_open() << '\n';
		return;
	}
	else
	{
		std::cerr << "data and labels loaded sucessfully\n";
	}
	dataFstream.seekg(16);
	labelsFstream.seekg(8);
	trainingData = fetchData(dataFstream, dimensions.at(0), trainingSetSize);
	trainingLabels = fetchLabels(labelsFstream, trainingSetSize);
}
train::~train()
{
}
double *train::fetchData(std::ifstream & data, int itemSize, int itemCount)
{
	unsigned char *c = new unsigned char[itemSize * itemCount];
	data.read(reinterpret_cast<char *> (c), itemSize * itemCount);

	double *output = new double[itemSize*itemCount];
	for (int i = 0; i < itemSize * itemCount; i++)
	{
			output[i] = c[i];
	}
	delete[] c;
	std::cout << "data fetched\n";
	return output;
}
int *train::fetchLabels(std::ifstream &labels, int itemCount)
{
	int *output = new int[itemCount];
	char label;
	for (int i = 0; i < itemCount; i++)
	{
		labels.read(&label, 1);
		output[i] = label;
	}
	std::cout << "labels fetched\n";
	return output;
}
double train::printHitrateInRange(int start, int end)
{
	auto startTime = Clock::now();
	Eigen::Index maxIndex;
	Eigen::VectorXd input;
	Eigen::VectorXd expectedOutput(10);
	Eigen::VectorXd output;
	int hit = 0;
	for (int i = start; i < end; i++)
	{
		input = Eigen::Map<Eigen::VectorXd>(trainingData + i * 28 * 28, 28 * 28);
		expectedOutput.setZero();
		expectedOutput(trainingLabels[i]) = 1;
		output = network.run(input, expectedOutput);
		output.maxCoeff(&maxIndex);
		if (trainingLabels[i] == maxIndex)
		{
			hit++;
		}
	}
	std::cout << "range:(" << start << "," << end << ")" << "\thitrate: " << hit << "/" << end-start+1 << "=" << (double)hit / (end-start) << '\n';
	std::cout << "average error:\t" << network.getAverageLastLayerError().transpose().sum()/10 << '\n';
	network.resetAverageLastLayerError();
	std::cout << "testing took "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - startTime).count() / 1e9
		<< " seconds\n";
	return (double)hit / (end - start + 1);
}
void train::start(int runs)
{
	const int checkingPeriod = 60000;
	//std::cout << "before training:\n";
	//printHitrateInRange(0, trainingSetSize-1);
	for (int i = 0; i < runs; i++)
	{
		auto runStart = Clock::now();
		auto batchStart = Clock::now();
		for (int j = 0; j < trainingSetSize; j++)
		{
			if (j % 6000 == 0)
			{
				std::cout << 100.0*j / 60000 << "%...";
			}
			Eigen::VectorXd expectedOutput(10);
			expectedOutput.setZero();
			expectedOutput(trainingLabels[j]) = 1;
			Eigen::VectorXd input = Eigen::Map<Eigen::VectorXd>(trainingData+j*28*28, 28*28);
			network.backpropagate(input, expectedOutput, j == (trainingSetSize-1));
		}
		std::cout << "\n\nrun " << i << " took "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - runStart).count() / 1e9
			<< " seconds\n";
		printHitrateInRange(0, trainingSetSize - 1);
		runStart = Clock::now();
	}
}