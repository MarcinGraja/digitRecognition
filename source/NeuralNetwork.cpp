#include <vector>
#include "NeuralNetwork.h"
#include <cmath>
#include <Eigen/StdVector>
#include <iostream>
#include <string>
void sigmoid(double x)
{
	x = 1.0 / (1 + std::exp(x));
}
void sigmoidDerivative(double x)
{
	x =  x * (1 - x);
}
Eigen::MatrixXd printDimensions(Eigen::MatrixXd m, std::string message)
{
	std::cout << message << " rows: " << m.rows() << " columns: " << m.cols() << '\n';
	return m;
}
void printDimensions(std::vector<Eigen::MatrixXd> v, std::string message)
{
	std::cout << message << '\n';
	for (auto m : v)
	{
		std::cout << " rows: " << m.rows() << " columns: " << m.cols() << '\n';
	}
}
NeuralNetwork::NeuralNetwork(const std::vector <int> &dimensions) : activations(0), activationsDerivatives(0), error(1), weights(1)
{
	for (int i = 0; i < dimensions.size()-1; i++)
	{
		weights.push_back(Eigen::MatrixXd::Random(dimensions.at(i+1), dimensions.at(i)));
	}
	activations.resize(dimensions.size());
	activationsDerivatives.resize(dimensions.size());
	error.resize(dimensions.size() - 1);
}
Eigen::VectorXd NeuralNetwork::run(const Eigen::Ref <Eigen::MatrixXd> &in, const Eigen::VectorXd &expectedOutput)
{
	int networkSize = activations.size();
	double learningRate = 1;
	activations.at(0) = in;
	activations.at(0).unaryExpr(&sigmoid);
	for (int i = 1; i <= weights.size(); i++)
	{
		std::vector <Eigen::MatrixXd> v;
		v.push_back(weights.at(i));
		v.push_back(activations.at(i-1));
		printDimensions(v, "weights, activations at" + std::to_string(i));
		activations.at(i) = weights.at(i) * activations.at(i - 1);
		activations.at(i).unaryExpr(&sigmoid);
	}
	for (int i = 0; i < activations.size(); i++)
	{
		activationsDerivatives.at(i) = activations.at(i);
		activationsDerivatives.at(i).unaryExpr(&sigmoidDerivative);
	}
	error.at(networkSize-1) = expectedOutput - activations.back();
	for (int i = networkSize-2 ; i >= 0; i--)
	{
		error.at(i) = printDimensions(weights.at(i+1).transpose(), "generating error weights") * printDimensions(error.at(i + 1), "generating error next error") * printDimensions(activationsDerivatives.at(i), "generating error activations derivatives");
		printDimensions(error.at(i), "error " + std::to_string(i));
	}
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights.at(i).cols(); j++)
			weights.at(i).col(j) -= (learningRate * error.at(i).array() * activations.at(i+1).array()).matrix();
		
	}
//	std::cout << "weights:\n" << weights.at(1) << '\n';
	return activations.back();
}
std::vector <Eigen::Matrix<double, -1, -1>, Eigen::aligned_allocator<Eigen::Matrix<double, -1, -1>>> NeuralNetwork::getWeights()
{
	return weights.getVector();
}
NeuralNetwork::~NeuralNetwork() {}