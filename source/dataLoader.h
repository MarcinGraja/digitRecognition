#pragma once
#include <fstream>
namespace dataLoader
{
	static void reset(std::ifstream data, std::ifstream labels)
	{
		data.seekg(16);
		labels.seekg(8);
	}
	static unsigned char *readData(std::ifstream data, int count)
	{
		char *value = new char[count];
		data.read(value, count);
		return reinterpret_cast<unsigned char*> (value);
	}
	static int readLabel(std::ifstream labels)
	{
		char value;
		labels.read(&value, 1);
		return value;
	}
};