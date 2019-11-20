#pragma once
#include <Eigen/core>
#include <vector>
class Adam
{
	std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> firstMoment;
	std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> secondMoment;
	std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> learningRates;
	double alpha;
	double decay1;
	double biasCorrectionArg1;
	double biasCorrectionArg2;
	double decay2;
	double epsilon;
public:
	std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> getLearningRates();
	void update(std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>& derivatives);
	Adam(const std::vector<int>& dimensions, double alpha = 0.003, double b1 = 0.9, double b2 = 0.999, double epsilon = 1e-8);
	~Adam();
};

