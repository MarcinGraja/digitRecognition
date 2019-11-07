#include <iostream>
#include <string>
#include <vector>
#include "train.h"
//image source http://yann.lecun.com/exdb/mnist/
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
		runs = 1000000;
		std::cout << "runs:\n";
		//std::cin >> runs;
		std::cout << runs;
		std::cout << "\n";
		trainer.start(runs);
	}
}
