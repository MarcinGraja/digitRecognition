#pragma once
#include <ctime>
#include <random>
#include "NeuralNet.h"
#include "train.h"
#include <iostream>
#include <vector>
#include <fstream>

void train::start(int runs, int netsInGeneration)
{
	int trainingSetSize = 60000;
	std::vector <neuralNet> nets;
	nets.reserve(netsInGeneration);
	std::ifstream dataFstream("train-images.idx3-ubyte", std::fstream::binary);
	std::ifstream labelsFstream("train-labels.idx1-ubyte", std::fstream::binary);
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
				output = nets[k].run(data);
				double max = output[0];
				double maxIndex = 0;
		}
	}
	delete data;
}