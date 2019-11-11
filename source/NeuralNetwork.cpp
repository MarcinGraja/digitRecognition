#include <vector>
#include "NeuralNetwork.h"
#include <cmath>
#include <Eigen/StdVector>
#include <iostream>
#include <string>
double sigmoid(double x)
{
	return 1.0 / (1 + std::exp(x));
}
double sigmoidDerivative(double x)
{
	return x * (1 - x);
}
void NeuralNetwork::backPropagate(double error)
{

}
Eigen::MatrixXd printDimensions(Eigen::MatrixXd m, std::string message)
{
	return m;
	std::cout << message << " rows: " << m.rows() << " columns: " << m.cols() << '\n';
	return m;
}
NeuralNetwork::NeuralNetwork(const std::vector <int> &dimensions)
{
	for (int i = 0; i < dimensions.size()-1; i++)
	{
		weights.push_back(Eigen::MatrixXd::Random(dimensions.at(i+1), dimensions.at(i)));
		std::string m = "weights " + std::to_string(i);
		printDimensions(weights.back(), m);
	}
}
Eigen::VectorXd NeuralNetwork::run(const Eigen::Ref <Eigen::MatrixXd> &in, const Eigen::VectorXd &expectedOutput)
{
	double learningRate = 0.5;
	std::vector <Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> activations;
	activations.push_back(in);
	activations.back().unaryExpr(&sigmoid);
	printDimensions(activations.back(), "activations " + std::to_string(0));
	for (int i = 0; i < weights.size(); i++)
	{
		activations.push_back((weights.at(i) * activations.back()).unaryExpr(&sigmoid));
		printDimensions(activations.back(), "activations " + std::to_string(i + 1));
	}
	std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> error;
	error.resize(activations.size() - 1);
	error.at(error.size()-1) = expectedOutput - activations.back();
	printDimensions(error.at(error.size() - 1), "error " + std::to_string(error.size() - 1));
	for (int i = error.size() - 2 ; i >= 0; i--)
	{
		error.at(i) = (weights.at(i + 1).transpose() * error.at(i + 1)).array() * activations.at(i+1).array();
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
	return weights;
}
