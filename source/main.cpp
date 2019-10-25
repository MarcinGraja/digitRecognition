#include <fstream>
#include <ctime>
#include <random>
#include <fstream>
#include <iostream>
#include <string>
#include "NeuralNet.h"
#include "train.h"
#include <vector>
//image source http://yann.lecun.com/exdb/mnist/
void debug(std::string Message)
{
	//std::cout << Message << '\n';
	return;
}


int main()
{
	std::vector<int> dimensions;
	dimensions.push_back(28 * 28);
	dimensions.push_back(100);
	dimensions.push_back(10);
	int runs = 1;
	train trainer(dimensions);
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
