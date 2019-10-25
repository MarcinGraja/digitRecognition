#pragma once
#include <ctime>
#include <random>
#include "NeuralNet.h"
#include "train.h"
#include <iostream>
#include <vector>
#include <fstream>

train::train(std::vector<int> dimensions)
{
	this->dimensions = dimensions;
}

train::~train()
{
}
std::vector <double> prepareData(unsigned char *data, int count)
{
	std::vector <double> v;
	v.reserve(count);
	for (int i = 0; i < count; i++)
	{
		v.push_back(data[i]);
	}
	return v;
}
void train::start(int runs, int netsInGeneration)
{
	int trainingSetSize = 60000;
	std::vector <neuralNet> nets;
	nets.reserve(netsInGeneration);
	std::ifstream dataFstream("data/train-images.idx3-ubyte", std::fstream::binary);
	std::ifstream labelsFstream("data/train-labels.idx1-ubyte", std::fstream::binary);
	if (!dataFstream.is_open() || !labelsFstream.is_open())
	{
		std::cerr << "data or labels not loaded; Data open:  " << dataFstream.is_open() << "labels open: " << labelsFstream.is_open() << '\n';
		return;
	}
	unsigned char *data = new unsigned char[dimensions.at(0)];
	for (int i = 0; i < runs; i++)
	{
		dataFstream.seekg(16);
		labelsFstream.seekg(8);
		for (int j = 0; j < trainingSetSize; j++)
		{
			dataFstream.read(reinterpret_cast<char *> (data), dimensions.at(0));
			char label;
			labelsFstream.read(&label, 1);
			for (int k = 0; k < netsInGeneration; k++)
			{
				std::vector <double> output;
				output = nets[k].run(prepareData(data, dimensions.at(0)));
				double max = output[0];
				double maxIndex = 0;
			}
		}
	}
	delete data;
}