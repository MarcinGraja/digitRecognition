#include <fstream>
#include "neuralNet.h"
#include "neuralLayer.h"
#include <vector>
struct vectorTuple
{
	std::vector <int> x;
	std::vector <int> y;
};
class neuralNet
{
	std::vector <neuralLayer> layers;
public:
	neuralNet(const neuralNet &other) :
	{
		
	}
	neuralNet(){};
	neuralNet(vectorTuple dimensions)
	{
		if (dimensions.x.size() != dimensions.y.size() || dimensions.x.size() < 2)
		{
			throw 
		}
	}
	~neuralNet()
	{
	}
	double *run(unsigned char *data)
	{
		for (int i = 0; i < inputRows; i++)
		{
			inputLayer[i].setValue((data[i]));
		}
		for (int i = 0; i < hiddenLayerColumns; i++)
		{
			for (int j = 0; j < hiddenLayerRows; j++)
			{
				hiddenLayer[i][j].updateValue(i == 0 ? inputLayer : hiddenLayer[i - 1]);
			}
		}
		for (int i = 0; i < outputRows; i++)
		{
			outputLayer[i].updateValue(hiddenLayer[hiddenLayerColumns - 1]);
		}
		double *returned = new double[outputRows] {0};
		int maxIndex = 0;
		double maxValue = outputLayer[0].getValue();
		double sum = 0;
		for (int i = 0; i < outputRows; i++)
		{
			returned[i] = outputLayer[i].getValue();
			returned[i] > 0 ? returned[i] : 0;
			sum += returned[i];
		}
		if (sum > 0)
			for (int i = 0; i < outputRows; i++)
			{
				returned[i] /= sum;
				if (returned[i] < 0)
					continue;
			}
		reset();
		return returned;
	}
	void randomizeMultipliers()
	{
		for (int i = 0; i < hiddenLayerColumns; i++)
		{
			for (int j = 0; j < hiddenLayerRows; j++)
			{
				hiddenLayer[i][j].updateMultipliers(5);
			}
		}
		for (int i = 0; i < outputRows; i++)
		{
			outputLayer[i].updateMultipliers(5);
		}
	}
};
neuralNet::neuralNet()
{
}


neuralNet::~neuralNet()
{
}
