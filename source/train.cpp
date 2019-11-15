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
	for (int i = 0; i < itemCount; i++)
	{
		std::vector <double> v;
		v.reserve(itemSize);
		for (int j = 0; j < itemSize; j++)
		{
			output[i] = (c[i * itemSize + j]-128) / 128.0;
		}
	}
	delete[] c;
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
	return output;
}
void train::printHitrateInRange(int start, int end)
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
	std::cout << "range:(" << start << "," << end << ")" << "\thitrate: " << hit << "/" << end-start << "=" << (double)hit / (end-start) << '\n';
	std::cout << network.getAverageLastLayerError().transpose() << '\n';
	network.resetAverageLastLayerError();
	std::cout << "testing took "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - startTime).count() / 1e9
		<< " seconds\n";
}
void train::start(int runs)
{
	const int checkingPeriod = 10000;
	for (int i = 0; i < runs; i++)
	{
		auto runStart = Clock::now();
		auto batchStart = Clock::now();
		for (int j = 0; j < trainingSetSize; j++)
		{
			Eigen::VectorXd expectedOutput(10);
			expectedOutput.setZero();
			expectedOutput(trainingLabels[j]) = 1;
			Eigen::VectorXd input = Eigen::Map<Eigen::VectorXd>(trainingData+j*28*28, 28*28);
			network.backpropagate(input, expectedOutput);
			if (NOTIFY_FREQUENTLY && (j + 1) % checkingPeriod == 0)
			{
				auto batchEnd = Clock::now();
				printHitrateInRange(j - checkingPeriod + 1, j);
				std::cout << "batch took " 
					<< std::chrono::duration_cast<std::chrono::nanoseconds>(batchEnd - batchStart).count() / 1e9 << "seconds\n\n";
				batchStart = Clock::now();
				
			}
		}
		std::cout << "run " << i << " took "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - runStart).count() / 1e9
			<< " seconds\n\n";
		if (i%10) printHitrateInRange(0, trainingSetSize);
		runStart = Clock::now();
	}
}