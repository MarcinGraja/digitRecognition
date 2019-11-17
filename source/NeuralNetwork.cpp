#include <vector>
#include "NeuralNetwork.h"
#include <cmath>
#include <Eigen/StdVector>
#include <iostream>
#include <string>
#include <iomanip>
double sigmoid(const double &x)
{
	//return x > 0 ? x : 0.01*x;
	return 1.0 / (1 + std::exp(-x));
}
double sigmoidDerivative(const double &x)
{
	//return x > 0? 1 : 0.01;
	return  x * (1 - x);
}
NeuralNetwork::NeuralNetwork(const std::vector <int> &dimensions, double startingLearningRate)
	: initialLearningRate(startingLearningRate),activations(0), activationsDerivatives(0), error(1), weights(1)
{
	learningRate = initialLearningRate;
	for (int i = 0; i < dimensions.size()-1; i++)
	{
		weights.push_back(Eigen::MatrixXd::Random(dimensions.at(i+1), dimensions.at(i)));
	}
	activations.resize(dimensions.size());
	activationsDerivatives.resize(dimensions.size());
	error.resize(dimensions.size() - 1);
	averageLastLayerError = Eigen::VectorXd(dimensions.back());
	averageLastLayerError.setZero();
}
Eigen::VectorXd NeuralNetwork::run(const Eigen::Ref <Eigen::MatrixXd> &in, const Eigen::VectorXd &expectedOutput)
{
	activations.at(0) = in;
	for (int i = 1; i <= weights.size(); i++)
	{
		activations.at(i) = (weights.at(i) * activations.at(i - 1)).unaryExpr(&sigmoid);
	}
	averageLastLayerError += (expectedOutput - activations.back()).array().square().matrix();
	errorCount++;
	return activations.back();
}
Eigen::VectorXd NeuralNetwork::backpropagate(const Eigen::Ref <Eigen::MatrixXd> &in, const Eigen::VectorXd &expectedOutput, bool debug)
{
	int networkSize = activations.size();
	activations.at(0) = in;
	for (int i = 1; i <= weights.size(); i++)
	{
		activations.at(i) = (weights.at(i) * activations.at(i - 1)).unaryExpr(&sigmoid);
	}
	for (int i = 0; i < activations.size(); i++)
	{
		activationsDerivatives.at(i) = activations.at(i).unaryExpr(&sigmoidDerivative);
	}
	error.back() = (activations.back() - expectedOutput).cwiseProduct(activationsDerivatives.back());
	for (int i = networkSize-2 ; i > 0; i--)
	{
		//cwiseProduct- multiplies corresponding elements from matrices
		error.at(i) = activationsDerivatives.at(i).cwiseProduct(weights.at(i + 1).transpose()* error.at(i + 1));	
	}
	for (int i = 1; i <= weights.size(); i++)
	{
		weights.at(i) -= (learningRate * error.at(i)) * activations.at(i - 1).transpose();
		if (i == weights.size())
		{
			//auto delta = learningRate * error.at(i) * activations.at(i - 1).transpose();
			//std::cout << delta.transpose();
			//std::cin.ignore();
		}
	}
	return activations.back();
}
std::vector <Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> NeuralNetwork::getWeights()
{
	return weights.getVector();
}
Eigen::VectorXd NeuralNetwork::getAverageLastLayerError()
{
	return averageLastLayerError/errorCount;
}
NeuralNetwork::~NeuralNetwork() {}

void NeuralNetwork::resetAverageLastLayerError()
{
	averageLastLayerError.setZero();
	errorCount = 0;
}
