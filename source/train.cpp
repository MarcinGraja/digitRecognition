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
typedef std::chrono::high_resolution_clock Clock;
train::train(std::vector<int> dimensions)
{
	this->dimensions = dimensions;
	network = NeuralNetwork(dimensions);
	trainingSetSize = 60000;
	std::ifstream dataFstream("data/train-images.idx3-ubyte", std::fstream::binary);
	std::ifstream labelsFstream("data/train-labels.idx1-ubyte", std::fstream::binary);
	if (!dataFstream.is_open() || !labelsFstream.is_open())
	{
		std::cerr << "data or labels not loaded; Data open:  " << dataFstream.is_open() << "labels open: " << labelsFstream.is_open() << '\n';
		return;
	}
	dataFstream.seekg(16);
	labelsFstream.seekg(8);
	data = fetchData(dataFstream, dimensions.at(0), trainingSetSize);
	labels = fetchLabels(labelsFstream, trainingSetSize);
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
	Eigen::VectorXd input = Eigen::Map<Eigen::Matrix<double, 28 * 28, 1>>(data + (j - k) * 28 * 28, 28 * 28);
	Eigen::VectorXd expectedOutput(10);
	expectedOutput.setZero();
	expectedOutput(labels[j - k]) = 1;
	Eigen::VectorXd output = network.run(input, expectedOutput);
	output.maxCoeff(&maxIndex);
	int hit = 0;

	for (int i = start; i < end; i++)
	{
		if (labels[j - k] == maxIndex)
		{
			hit++;
		}
	}
	std::cout << "range:(" << start << "," << end << ")" << "\thitrate: " << hit << "/" << end-start << "=" << (double)hit / (end-start) << '\n';
	std::cout << "hitcount:" << hit << '\n';
	auto currentUpdate = Clock::now();
	std::cout << network.getAverageLastLayerError() << '\n';
	network.resetAverageLastLayerError();
	auto t2 = Clock::now();
	std::cout << "time of batch: "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(currentUpdate - Clock::now()).count() / 1e9
		<< " seconds" << std::endl;
}
void train::start(int runs)
{
	
	const int checkingPeriod = 2;
	const int sameImageRepeat = 100;
	int hit = 0;

	for (int i = 0; i < runs; i++)
	{
		for (int j = 0; j < trainingSetSize; j++)
		{
			Eigen::VectorXd expectedOutput(10);
			expectedOutput.setZero();
			expectedOutput(labels[j]) = 1;
			Eigen::VectorXd input = Eigen::Map<Eigen::Matrix<double, 28*28, 1>>(data+j*28*28, 28*28);
			Eigen::VectorXd output;
			for (int k = 0; k < sameImageRepeat; k++)
			{
				output = network.backpropagate(input, expectedOutput);
				
			}
			if ((j + 1) % checkingPeriod == 0)
			{
				printHitrateInRange(j - checkingPeriod, j);
				break;
			}
		}
		int totalHit = 0;
		//for (int j = 0; j < trainingSetSize; j++)
		//{
		//	Eigen::VectorXd input = Eigen::Map<Eigen::Matrix<double, 28 * 28, 1>>(data + j * 28 * 28, 28 * 28);
		//	Eigen::VectorXd output;
		//	output = network.run(input);
		//	Eigen::Index maxIndex;
		//	output.maxCoeff(&maxIndex);
		//	if (labels[j] == maxIndex)
		//	{
		//		hit++;
		//		totalHit++;
		//	}
		//}
		//std::cout << "time of run: "
		//	<< std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - runStart).count() / 1e9
		//	<< " seconds" << std::endl;
		//std::cout << "total hit rate:" << (double)totalHit / trainingSetSize << '\n';
		//lastUpdate = Clock::now();
	}
}