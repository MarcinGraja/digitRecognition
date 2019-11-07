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
	int checkingPeriod = 1000;
	for (int i = 0; i < runs; i++)
	{
		for (int j = 0; j < trainingSetSize; j++)
		{
			if (j % checkingPeriod == 0 && j != 0)
			{
				std::cout << "current image: " << j << "\thitrate: " << hit / checkingPeriod << '\n';
				hit = 0;
			}
			Eigen::RowVectorXd expectedOutput(10);
			expectedOutput.setZero();
			expectedOutput(labels[j]) = 1;
			Eigen::MatrixXd input = Eigen::Map<Eigen::Matrix<double, 1, 28*28>>(data, 28*28);
			network.run(input, expectedOutput);
		}
	}
}