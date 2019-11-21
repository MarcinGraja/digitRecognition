#pragma once
#include <fstream>
#include <iostream>
namespace dataLoader 
{
	double *fetchData(std::ifstream & data, int itemSize, int itemCount)
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
	int *fetchLabels(std::ifstream &labels, int itemCount)
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
	void fetchDataAndLabels(double *&trainingData, int *&trainingLabels, double *&testingData, int *&testingLabels, int trainingSetSize, int testingSetSize)
	{
		std::ifstream trainingDataFstream("data/train-images.idx3-ubyte", std::fstream::binary);
		std::ifstream trainingLabelsFstream("data/train-labels.idx1-ubyte", std::fstream::binary);
		std::ifstream testingDataFstream("data/t10k-images.idx3-ubyte", std::fstream::binary);
		std::ifstream testingLabelsFstream("data/t10k-labels.idx1-ubyte", std::fstream::binary);
		if (!trainingDataFstream.is_open() || !trainingLabelsFstream.is_open() 
			|| !testingDataFstream.is_open() || ! testingLabelsFstream.is_open())
		{
			std::cerr << "data or labels not loaded; training data open:  " << trainingDataFstream.is_open() 
				<< " training labels open: " << trainingLabelsFstream.is_open() 
				<< " testing data open:" << testingDataFstream.is_open() 
				<< " testing labels open: " <<  testingLabelsFstream.is_open() <<'\n';
			return;
		}
		else
		{
			std::cerr << "data and labels loaded sucessfully\n";
		}
		trainingDataFstream.seekg(16);
		trainingLabelsFstream.seekg(8);
		testingDataFstream.seekg(16);
		testingLabelsFstream.seekg(8);
		trainingData = fetchData(trainingDataFstream, 28*28, trainingSetSize);
		trainingLabels = fetchLabels(trainingLabelsFstream, trainingSetSize);
		testingData = fetchData(testingDataFstream, 28 * 28, testingSetSize);
		testingLabels = fetchLabels(testingLabelsFstream, testingSetSize);
	}

}