#include <iostream>
#include <string>
#include <vector>
#include "train.h"
//image source http://yann.lecun.com/exdb/mnist/
int main()
{
	std::vector<int> dimensions;
	dimensions.push_back(28 * 28);
	dimensions.push_back(128);
	dimensions.push_back(10);
	int runs = 1000;
	train trainer(dimensions, 60000, 10000);
	std::cout << "runs:\n";
	std::cout << runs;
	std::cout << "\n";
	trainer.run(runs, 5);
	std::cout << "finito";
	while (true)
	{
		std::cin.ignore();
	}
}
