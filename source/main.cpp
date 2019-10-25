#include <fstream>
#include <ctime>
#include <random>
#include <fstream>
#include <iostream>
#include <string>
#include "NeuralNet.h"
#include "train.h"
//image source http://yann.lecun.com/exdb/mnist/
void debug(std::string Message)
{
	//std::cout << Message << '\n';
	return;
}


int main()
{
	const int inputRows = 28 * 28;
	const int hiddenLayerRows = 10;
	const int hiddenLayerColumns = 1;
	const int outputRows = 10;
	int runs = 1;
	train trainer;
	trainer.initialize(inputRows, hiddenLayerRows, hiddenLayerColumns, outputRows);
	while (runs)
	{
		int netsInGeneration;
		std::cout << "runs, nets in generation:\n";
		//std::cin >> runs;
		std::cout << "\n";
		//std::cin >> netsInGeneration;
		std::cout << "\n";
		runs = 10000; netsInGeneration = 2;
		trainer.start(runs, netsInGeneration);
	}
}
