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
		vectorHandler &operator=(const vectorHandler &other)
		{
			this->v = other.v;
			this->offset = other.offset;
			return *this;
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
	NeuralNetwork &operator=(const NeuralNetwork &other)
	{
		this->activations = other.activations;
		this->activationsDerivatives = other.activationsDerivatives;
		this->error = other.error;
		this->weights = other.weights;
		this->averageLastLayerError = other.averageLastLayerError;
		this->learningRate = other.learningRate;
		return *this;
	}
	void updateLearningRate(double base, double step, int x)
	{
		learningRate = base * std::exp(-step*x);
		//std::cout << "learning rate set to \t" << learningRate << '\n';
	}
	double getLearningRate()
	{
		return learningRate;
	}
};