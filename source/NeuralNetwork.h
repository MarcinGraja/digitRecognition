#pragma once
#include <Eigen/core>
#include <vector>
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
			return v.at(index - offset);
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
	int errorCount = 0;
public:
	NeuralNetwork(const std::vector<int>& dimensions);
	Eigen::VectorXd run(const Eigen::Ref<Eigen::MatrixXd>& in, const Eigen::VectorXd & expectedOutput);
	std::vector<Eigen::Matrix<double, -1, -1>, Eigen::aligned_allocator<Eigen::Matrix<double, -1, -1>>> getWeights();
	Eigen::VectorXd getAverageLastLayerError();
	Eigen::VectorXd backpropagate(const Eigen::Ref<Eigen::MatrixXd>& in, const Eigen::VectorXd & expectedOutput);
	~NeuralNetwork();
	void resetAverageLastLayerError();
};