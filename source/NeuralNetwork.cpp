#include <vector>
#include "NeuralNetwork.h"
#include <cmath>
#include <Eigen/StdVector>
#include <iostream>
#include <string>
double sigmoid(const double &x)
{
	return 1.0 / (1 + std::exp(x));
}
double sigmoidDerivative(const double &x)
{
	return  x * (1 - x);
}
Eigen::MatrixXd printDimensions(Eigen::MatrixXd m, std::string message)
{
	return m;
//	std::cout << message << " rows: " << m.rows() << " columns: " << m.cols() << '\n';
	return m;
}
void printDimensions(std::vector<Eigen::MatrixXd> v, std::string message)
{
	return;
//	std::cout << message << '\n';
	for (auto m : v)
	{
//		std::cout << " rows: " << m.rows() << " columns: " << m.cols() << '\n';
	}
}
NeuralNetwork::NeuralNetwork(const std::vector <int> &dimensions) : activations(0), activationsDerivatives(0)/**/, error(1), weights(1)
{
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
	int networkSize = activations.size();
	double learningRate = 0.1;
	activations.at(0) = in;
	activations.at(0) = activations.at(0).unaryExpr(&sigmoid);
	for (int i = 1; i <= weights.size(); i++)
	{
		std::vector <Eigen::MatrixXd> v;
		v.push_back(weights.at(i));
		v.push_back(activations.at(i - 1));
		printDimensions(v, "weights, activations at" + std::to_string(i));
		activations.at(i) = weights.at(i) * activations.at(i - 1);
		activations.at(i) = activations.at(i).unaryExpr(&sigmoid);
	}
	averageLastLayerError += (expectedOutput - activations.back()).array().square().matrix();
	errorCount++;
	return activations.back();
}
Eigen::VectorXd NeuralNetwork::backpropagate(const Eigen::Ref <Eigen::MatrixXd> &in, const Eigen::VectorXd &expectedOutput)
{
	int networkSize = activations.size();
	double learningRate = 0.03;
	activations.at(0) = in;
	activations.at(0) = activations.at(0).unaryExpr(&sigmoid);
	for (int i = 1; i <= weights.size(); i++)
	{
		std::vector <Eigen::MatrixXd> v;
		v.push_back(weights.at(i));
		v.push_back(activations.at(i-1));
		printDimensions(v, "weights, activations at" + std::to_string(i));
		activations.at(i) = weights.at(i) * activations.at(i - 1);
		activations.at(i)= activations.at(i).unaryExpr(&sigmoid);
//		std::cout << i << ":\n" << activations.at(i) << '\n';
	}
	for (int i = 0; i < activations.size(); i++)
	{
		activationsDerivatives.at(i) = activations.at(i);
		activationsDerivatives.at(i) = activationsDerivatives.at(i).unaryExpr(&sigmoidDerivative);
	}/**/
	error.at(networkSize-1) = expectedOutput - activations.back();
	for (int i = networkSize-2 ; i > 0; i--)
	{
//		error.at(i) = printDimensions(weights.at(i+1).transpose(), "generating error weights") * printDimensions(error.at(i + 1), "generating error next error") * printDimensions(activationsDerivatives.at(i), "generating error activations derivatives");
		error.at(i) = activationsDerivatives.at(i).cwiseProduct(weights.at(i + 1).transpose()* error.at(i + 1));
//		error.at(i) = weights.at(i + 1).transpose()* error.at(i + 1);
		//printDimensions(error.at(i), "error " + std::to_string(i));
	}/**/
//	std::cout << "weights:\n" << weights.at(2).col(0) << '\n';
//	std::cout << "error:\n" << error.at(2);
	for (int i = 1; i <= weights.size(); i++)
	{
		auto delta = learningRate * error.at(i) * activations.at(i - 1).transpose();
		weights.at(i) -= delta;
		if (i == weights.size() && false)
		{
			std::cout << delta.col(0) << '\n';
			std::cin.ignore();
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
