#pragma once
#include <Eigen/core>
#include <vector>
#include <iostream>
#include <cmath>
#define type Eigen::MatrixXd
#define allocator Eigen::aligned_allocator<Eigen::MatrixXd>
class NeuralNetwork
{	
	#define type Eigen::MatrixXd
	#define allocator Eigen::aligned_allocator<Eigen::MatrixXd>
	class vectorHandler
	{
		int offset;
		std::vector<type, allocator> v;
	public:
		vectorHandler(int offset)
		{
			this->offset = offset;
		}
		type &at(int index)
		{
			return v.operator[](index - offset);
		}
		type &operator[](int index)
		{
			return v.operator[](index - offset);
		}
		void resize(int size)
		{
			v.resize(size);
		}
		void push_back(type item)
		{
			v.push_back(item);
		}
		type &back()
		{
			return v.back();
		}
		int size()
		{
			return v.size();
		}
		std::vector<type, allocator> &getVector()
		{
			return v;
		}
		
	};
	#undef type
	#undef allocator
	vectorHandler activations;
	vectorHandler activationsDerivatives;
	vectorHandler error;
	vectorHandler weights;
	Eigen::VectorXd averageLastLayerError;
	const double initialLearningRate;
	double learningRate;
	int errorCount = 0;
public:
	NeuralNetwork(const std::vector<int>& dimensions, double startingLearningRate = 0.03);
	Eigen::VectorXd run(const Eigen::Ref<Eigen::MatrixXd>& in, const Eigen::VectorXd & expectedOutput);
	std::vector<Eigen::Matrix<double, -1, -1>, Eigen::aligned_allocator<Eigen::Matrix<double, -1, -1>>> getWeights();
	Eigen::VectorXd getAverageLastLayerError();
	Eigen::VectorXd backpropagate(const Eigen::Ref<Eigen::MatrixXd>& in, const Eigen::VectorXd & expectedOutput, bool debug = false);
	~NeuralNetwork();
	void resetAverageLastLayerError();
	void updateLearningRate(int run)
	{
		learningRate = 0.01 * std::exp(-0.05*run);
		std::cout << "learning rate set to \t" << learningRate << '\n';
	}
};