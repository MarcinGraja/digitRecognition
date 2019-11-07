#include <vector>
#include "NeuralNetwork.h"
#include <cmath>
#include <Eigen/StdVector>
#include <iostream>
double sigmoid(double x)
{
	return 1.0 / (1 + std::exp(x));
}
void NeuralNetwork::backPropagate(double error)
{

}
NeuralNetwork::NeuralNetwork(const std::vector <int> &dimensions)
{
	for (int i = 0; i < dimensions.size()-1; i++)
	{
		weights.push_back(Eigen::MatrixXd::Random(dimensions.at(i), dimensions.at(i+1)));
	}
}

void NeuralNetwork::run(const Eigen::Ref <Eigen::MatrixXd> &in, const Eigen::RowVectorXd &expectedOutput)
{
	double learningRate = 0.1;
	std::vector <Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> activations;
	activations.push_back(in);
	activations.back().unaryExpr(&sigmoid);
	for (int i = 0; i < weights.size(); i++)
	{
		activations.push_back((activations.back() * weights.at(i)).unaryExpr(&sigmoid));
	}
	std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> error;
	error.resize(activations.size() - 1);
	error.at(error.size()-1) = expectedOutput - activations.back();
	for (int i = error.size() - 2 ; i >= 0; i--)
	{

		error.at(i) = (weights.at(i + 1).transpose() * error.at(i + 1));
	}
	for (int i = 0; i < weights.size(); i++)
	{
		std::cout << "i:" << i << " " << weights.at(i).rows() << "x" << weights.at(i).cols() << '\t';
		std::cout << error.at(i).rows() << "x" << error.at(i).cols() << '\n';
		weights.at(i) -= learningRate * error.at(i);
	}
}
NeuralNetwork::~NeuralNetwork()
{
}
