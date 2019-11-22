#include <iostream>
#include <string>
#include <vector>
#include "train.h"
//image source http://yann.lecun.com/exdb/mnist/
//database size: 0:5923  1:6742  2:5958  3:6131  4:5842  5:5421  6:5918  7:6265  8:5851  9:5949
int main()
{
	std::vector<int> dimensions;
	dimensions.push_back(28 * 28);
	dimensions.push_back(50);
	dimensions.push_back(10);
	int runs = 1000;
	train trainer(dimensions, 60000, 10000);
	std::cout << "runs:\n";
	std::cout << runs;
	std::cout << "\n";
	trainer.run(runs);
	std::cout << "finito";
	while (true)
	{
		std::cin.ignore();
	}
}
