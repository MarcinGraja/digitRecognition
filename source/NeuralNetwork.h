#pragma once
#include <Eigen/core>
class NeuralNetwork
{
	std::vector <Eigen::Matrix<double, -1, -1>, Eigen::aligned_allocator<Eigen::Matrix<double, -1, -1>>> weights;
	void backPropagate(double error);
public:
	NeuralNetwork(const std::vector<int>& dimensions);
	void run(const Eigen::Ref<Eigen::MatrixXd>& in, const Eigen::RowVectorXd & expectedOutput);
	~NeuralNetwork();
};