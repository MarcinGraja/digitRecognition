#pragma once
#include <iostream>
#include <cmath>
#include <algorithm>
#include "NeuralNetwork.h"
#include "train.h"
#include "Eigen/core"
#include <chrono>
#include <string>
#include "dataLoader.h"
#include <thread>
typedef std::chrono::high_resolution_clock Clock;
train::train(std::vector<int> dimensions, int trainingSetSize, int testingSetSize)
{
	log.open("log.txt", std::ofstream::trunc);
	if (!log.is_open())
	{
		std::cout << "log not opened\n";
	}
	else std::cout << "log opened\n";
	csvLog.open("log.csv", std::ofstream::trunc);
	if (!log.is_open())
	{
		std::cout << "csv log not opened\n";
	}
	else std::cout << "csv log opened\n";
	this->dimensions = dimensions;
	this->trainingSetSize = trainingSetSize;
	this->testingSetSize = testingSetSize;
	dataLoader::fetchDataAndLabels(trainingData, trainingLabels, testingData, testingLabels, trainingSetSize, testingSetSize);
}
train::~train()
{
}

double train::printHitrateInRange(int start, int end, NeuralNetwork network, std::string m, std::string csvM)
{
	auto startTime = Clock::now();
	Eigen::Index maxIndex;
	Eigen::VectorXd input;
	Eigen::VectorXd expectedOutput(10);
	Eigen::VectorXd output;
	int hit = 0;
	for (int i = start; i < end; i++)
	{
		input = Eigen::Map<Eigen::VectorXd>(testingData + i * 28 * 28, 28 * 28);
		expectedOutput.setZero();
		expectedOutput(testingLabels[i]) = 1;
		output = network.run(input, expectedOutput);
		output.maxCoeff(&maxIndex);
		if (testingLabels[i] == maxIndex)
		{
			hit++;
		}
	}
	double averageError = network.getAverageLastLayerError().sum() / 10;
	std::string message = m + ":range:(" + std::to_string(start) + "," + std::to_string(end)
		+ ")\thitrate: " + std::to_string(hit) + "/" + std::to_string(end - start + 1) + "=" 
		+ std::to_string((double)hit / (end - start + 1)) + "\n average error:\t" + std::to_string(averageError)
		+ "\n testing took " + std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - startTime).count() / 1e9)
		+ " seconds\n";
	std::cout << message;
	std::string csvMessage = csvM + std::to_string(averageError) + ";" + std::to_string((double)hit / (end - start + 1)) + '\n';
	log << message;
	csvLog << csvMessage;
	network.resetAverageLastLayerError();
	return averageError;
}
void train::run(int runs, int batch)
{
	double step = 0.407014;
	double base = 0.0814027;
	NeuralNetwork net(dimensions);
	net.updateLearningRate(base, step, 0);
	double dummy;
	for (int i = 0; i < runs; i++)
		backpropagate(net, base, step, batch, i*batch, dummy);
}
void train::findHyperParameters(int runs)
{

	std::cout << "learning rate update formula: learningRate = base * std::exp(-step*x);\n";
	std::cout << "\n\n\n";
	bool again = true;

	double bestBase = 0.01;
	double bestStep = 0.05;
	double metaLearningRate = 0.1;
	std::vector<std::thread> threads;
	std::vector<NeuralNetwork> nets;
	for (int k = 0; k < runs; k++)
	{
		auto runStart = Clock::now();
		threads.clear();
		nets.clear();
		double base[3] = { bestBase * (1 - metaLearningRate), bestBase , bestBase * (1 + metaLearningRate) };
		double step[3] = { bestStep * (1 - metaLearningRate), bestStep, bestStep * (1 + metaLearningRate) };
		for (int i = 0; i < 9; i++)
		{
			nets.push_back(NeuralNetwork(dimensions));
		}
		double *returned = new double[9];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				threads.push_back(std::thread(&train::backpropagate, this, std::ref(nets.at(i*3+j)), base[i], step[j], runs, 0, std::ref(returned[i*3+j])));
			}
		}
		double bestError = 1e300;
		int bestIndex;
		for (int i = 0; i < 9; i++)
		{
			threads.at(i).join();
			if (returned[i] < bestError) bestIndex = i;
		}
		bestBase = base[bestIndex / 3];
		bestStep = step[bestIndex % 3];
		std::cout << "best step:" << bestStep << " best base: " << bestBase;
		std::cout << "\n run took " << std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - runStart).count() / 1e9 <<'\n';
	}
	
}
void train::backpropagate(NeuralNetwork &network, double base, double step, int runs, int currentRun, double &returned)
{
	auto start = Clock::now();
	for (int j = 0; j < runs; j++)
	{
		network.updateLearningRate(base, step, j + currentRun);
		for (int i = 0; i < trainingSetSize; i++)
		{
			Eigen::VectorXd expectedOutput(10);
			expectedOutput.setZero();
			expectedOutput(trainingLabels[i]) = 1;
			Eigen::VectorXd input = Eigen::Map<Eigen::VectorXd>(trainingData + i * 28 * 28, 28 * 28);
			network.backpropagate(input, expectedOutput);
		}
	}
	std::string message = "current learning rate: " + std::to_string(network.getLearningRate()) + " base:\t " + std::to_string(base) + "\tstep:\t" + std::to_string(step) + '\n';
	std::string csvMessage = std::to_string(network.getLearningRate()) + ";" + std::to_string(base) + ";" + std::to_string(step) + ";";
	std::cout << runs << " runs took "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count() / 1e9 << '\n';
	returned = printHitrateInRange(0, testingSetSize - 1, network, message, csvMessage);
}
void train::probTrashChoosesOptimalLearningRate(double runs)
{
	double initialBase = 0.5, initialStep = 0.5;
	std::vector <std::thread> threads;
	std::vector <NeuralNetwork> nets;
	for (double base = initialBase; base > 0; base -= 0.1*initialBase)
	{
		for (double step = initialStep; step > 0; step -= 0.1*initialStep)
		{
			nets.push_back(NeuralNetwork(dimensions));
		}
	}
	int i = 0;
	for (double base = initialBase; base > 0; base -= 0.1*initialBase)
	{
		for (double step = initialStep; step > 0; step -= 0.1*initialStep)
		{
//			threads.push_back(std::thread(&train::backpropagate, this, std::ref(nets.at(i)), base, step, runs));
			i++;
		}
	}

	for (int i = 0; i < threads.size(); i++)
	{
		threads.at(i).join();
	}
}