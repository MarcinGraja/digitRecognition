#pragma once
#include <ctime>
#include <random>
#include "NeuralNet.h"
#include <iostream>
class train {
	int inputRows;
	int hiddenLayerRows;
	int hiddenLayerColumns;
	int outputRows;
public:
	void initialize(int inputRows, int hiddenLayerRows, int hiddenLayerColumns, int outputRows)
	{
		std::srand(std::time(NULL));
		this->inputRows = inputRows;
		this->hiddenLayerRows = hiddenLayerRows;
		this->hiddenLayerColumns = hiddenLayerColumns;
		this->outputRows = outputRows;
	}
	void start(int runs, int netsInGeneration)
	{
		int trainingSetSize = 60000;
		double* score = new double[netsInGeneration];
		for (int i = 0; i < netsInGeneration; i++)
		{
			score[i] = 0;
		}
		neuralNet **nets = new neuralNet*[netsInGeneration];
		nets[0] = new neuralNet(inputRows, hiddenLayerRows, hiddenLayerColumns, outputRows);
		int *guesses = new int[netsInGeneration];
		std::ifstream dataFstream("train-images.idx3-ubyte", std::fstream::binary);
		std::ifstream labelsFstream("train-labels.idx1-ubyte", std::fstream::binary);
		unsigned char *data = new unsigned char[inputRows];
		for (int i = 0; i < runs; i++)
		{
			dataFstream.seekg(16);
			labelsFstream.seekg(8);
			for (int j = 1; j < netsInGeneration; j++)
			{
				nets[j] = new neuralNet(*nets[0]);
				nets[j]->randomizeMultipliers();
			}
			for (int j = 0; j < trainingSetSize; j++)
			{
				if (j % (trainingSetSize/10) == 0)
				{
					std::cout << 100 * j / trainingSetSize << "%" << '\n';
					for (int k = 0; k < netsInGeneration; k++)
					{
						std::cout << "net:" << k << "\tscore:" << score[k] << '\n';
					}
				}
				dataFstream.read(reinterpret_cast<char *> (data), inputRows);
				char label;
				labelsFstream.read(&label, 1);
				for (int k = 0; k < netsInGeneration; k++)
				{
					double *output = new double[outputRows];
					output = nets[k]->run(data);
					score[k] += output[label];
					double max = output[0];
					double maxIndex = 0;
					for (int i = 1; i < outputRows; i++)
					{
						if (output[i] > max)
						{
							max = output[i];
							maxIndex = i;
						}
					}
					if (maxIndex == label)
						guesses[k]++;
				}
			}
			double maxScore = score[0];
			int maxScoreIndex = 0;
			for (int j = 0; j < netsInGeneration; j++)
			{
				std::cout << "net:" << j << "\tscore:" << score[j] << '\n';
				if (score[j] > maxScore)
				{
					maxScore = score[j];
					maxScoreIndex = j;
				}
			}
			std::cout << "run:\t" << i << "best net:" << maxScoreIndex <<" max score:" << maxScore << '\n';
			nets[0] = nets[maxScoreIndex];
			for (int i = 0; i < netsInGeneration; i++)
			{
				std::cout << "net " << i << "\t score:\t" << score[i] << "accuracy: " << guesses[i]/100.0 << "%" << '\n';
			}
			for (int j = 1; j < netsInGeneration; j++)
			{
				delete nets[j];
			}

		}
		delete data;
	}
};