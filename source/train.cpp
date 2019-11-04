#pragma once
#include <ctime>
#include <random>
#include "NeuralNet.h"
#include "train.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
train::train(std::vector<int> dimensions)
{
	this->dimensions = dimensions;
}

train::~train()
{
}
void fetchData(std::vector <std::vector<double>> &output, std::ifstream & data, int itemSize, int itemCount)
{
	unsigned char *c = new unsigned char[itemSize * itemCount];
	data.read(reinterpret_cast<char *> (c), itemSize * itemCount);

	output.reserve(itemCount);
	for (int i = 0; i < itemCount; i++)
	{
		std::vector <double> v;
		v.reserve(itemSize);
		for (int j = 0; j < itemSize; j++)
		{
			v.push_back((c[i * itemSize + j]-128) / 128.0);
		}
		output.push_back(v);
	}
	delete[] c;
}
void fetchLabels(std::vector <int> &output, std::ifstream &labels, int itemCount)
{
	output.reserve(itemCount);
	char label;
	for (int i = 0; i < itemCount; i++)
	{
		labels.read(&label, 1);
		output.push_back(label);
	}
}
void train::start(int runs)
{
	int trainingSetSize = 60000;
	std::ifstream dataFstream("data/train-images.idx3-ubyte", std::fstream::binary);
	std::ifstream labelsFstream("data/train-labels.idx1-ubyte", std::fstream::binary);
	dataFstream.seekg(16);
	labelsFstream.seekg(8);
	std::vector<std::vector<double>> data;
	fetchData(data, dataFstream, dimensions.at(0), trainingSetSize);
	std::vector <int> labels;
	fetchLabels(labels, labelsFstream, trainingSetSize);
	neuralNet network(dimensions);
	if (!dataFstream.is_open() || !labelsFstream.is_open())
	{
		std::cerr << "data or labels not loaded; Data open:  " << dataFstream.is_open() << "labels open: " << labelsFstream.is_open() << '\n';
		return;
	}
	int hit = 0;
	int batchSize = 100;
	std::vector <double> error;
	error.resize(dimensions.back());
	for (int i = 0; i < runs; i++)
	{
		for (int j = 0; j < 100; j++)
		{
			if ((j+1) % batchSize == 0)
			{
				double average = 0;
				for (double &d : error)
				{
					d /= batchSize;
					average += d;
				}
				average /= dimensions.back();
				network.backPropagate(error);
				std::cout << "run : " << i << "\timage: " << j << "\nhitrate:\t" << hit << "/" << batchSize << "=" << (double)hit / batchSize << '\n';
				std::cout << "Error:" << average << "\n";
				for (double d : error)
				{
					std::cout << d << '\t';
				}
				std::cout << '\n';
				hit = 0;
				error.clear();
				error.resize(dimensions.back(), 0);
			}
			std::vector <double> result = network.run(data.at(j));
			int maxIndex = 0;
			double max = result.at(0);
			for (int i = 0; i < result.size(); i++)
			{
				if (max < result.at(i))
				{
					max = result.at(i);
					maxIndex = i;
				}
			}
			hit += (maxIndex == labels.at(j));
			for (int i = 0; i < dimensions.back(); i++)
			{
				error.at(i) += (i == labels.at(j) ? std::pow(1 - result.at(i), 2) : std::pow(result.at(i), 2));
			}
		}
	}
}