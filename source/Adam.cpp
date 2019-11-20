#include "Adam.h"
#include <iostream>

Adam::Adam(const std::vector<int>& dimensions, double alpha, double b1, double b2, double epsilon)
{
	this->alpha = alpha;
	biasCorrectionArg1 = decay1 = b1;
	biasCorrectionArg2 = decay2 = b2;
	this->epsilon = epsilon;
	for (int i = 0; i < dimensions.size() - 1; i++)
	{
		auto d = Eigen::MatrixXd(dimensions.at(i + 1), dimensions.at(i));
		firstMoment.push_back(d);
		secondMoment.push_back(d);
		learningRates.push_back(d);
		firstMoment.back().fill(0);
		secondMoment.back().fill(0);
		learningRates.back().fill(0.01);
	}
}

Adam::~Adam()
{
}
std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> Adam::getLearningRates()
{
	std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> lr;
	for (Eigen::ArrayXXd arr : learningRates)
		lr.push_back(arr);
	return lr;
}
void Adam::update(std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> &weightsDerivatives)
{
	for (int i = 0; i < weightsDerivatives.size(); i++)
	{
		firstMoment.at(i) = decay1 * firstMoment.at(i) + (1 - decay1) * weightsDerivatives.at(i);
		secondMoment.at(i) = decay2 * secondMoment.at(i) + (1 - decay2) * (weightsDerivatives.at(i).cwiseProduct(weightsDerivatives.at(i)));
		Eigen::MatrixXd delta = alpha * (firstMoment.at(i) / (1 - biasCorrectionArg1)).array()
			/ ((secondMoment.at(i) / (1 - biasCorrectionArg2)).array().sqrt() + epsilon).array();

		learningRates.at(i) -= delta;
	}
	biasCorrectionArg1 *= decay1;
	biasCorrectionArg2 *= decay2;
}