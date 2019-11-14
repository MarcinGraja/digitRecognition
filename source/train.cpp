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
}

train::~train()
{
}
double *fetchData(std::ifstream & data, int itemSize, int itemCount)
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
int *fetchLabels(std::ifstream &labels, int itemCount)
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
void train::start(int runs)
{
	int trainingSetSize = 60000;
	std::ifstream dataFstream("data/train-images.idx3-ubyte", std::fstream::binary);
	std::ifstream labelsFstream("data/train-labels.idx1-ubyte", std::fstream::binary);
	if (!dataFstream.is_open() || !labelsFstream.is_open())
	{
		std::cerr << "data or labels not loaded; Data open:  " << dataFstream.is_open() << "labels open: " << labelsFstream.is_open() << '\n';
		return;
	}
	dataFstream.seekg(16);
	labelsFstream.seekg(8);
	
	double *data = fetchData(dataFstream, dimensions.at(0), trainingSetSize);
	int *labels = fetchLabels(labelsFstream, trainingSetSize);
	NeuralNetwork network(dimensions);
	int hit = 0;
	int checkingPeriod = 10000;
	
	
	for (int i = 0; i < runs; i++)
	{
		auto runStart = Clock::now();
		auto lastUpdate = Clock::now();
		int totalHit = 0;
		for (int j = 0; j < trainingSetSize; j++)
		{

			if ((j+1) % 10000 == 0 && false)
			{
				auto currentUpdate = Clock::now();
				std::cout << "current image: " << (j+1) << "\thitrate: " << (double)hit / checkingPeriod << '\n';
				hit = 0;
				std::cout << network.getWeights().at(1).col(0) << '\n';
				auto t2 = Clock::now();
				std::cout << "time of batch: "
					<< std::chrono::duration_cast<std::chrono::nanoseconds>(currentUpdate - lastUpdate).count() / 1e9
					<< " seconds" << std::endl;
				lastUpdate = Clock::now();
			}
			Eigen::VectorXd expectedOutput(10);
			expectedOutput.setZero();
			expectedOutput(labels[j]) = 1;
			Eigen::VectorXd input = Eigen::Map<Eigen::Matrix<double, 28*28, 1>>(data+j*28*28, 28*28);
			Eigen::VectorXd output = network.run(input, expectedOutput);
			Eigen::Index maxIndex;
			output.maxCoeff(&maxIndex);
			if (labels[j] == maxIndex)
			{
				hit++;
				totalHit++;
			}
		}
		std::cout << "time of run: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - runStart).count() / 1e9
			<< " seconds" << std::endl;
		std::cout << "total hit rate:" << (double)totalHit / trainingSetSize << '\n';
		lastUpdate = Clock::now();
	}
}