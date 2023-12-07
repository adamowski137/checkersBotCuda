#pragma once
#include "cuda_runtime.h"

template <typename T>
class DeviceVector 
{
private:
	size_t count;
public:
	T array[32];
	__device__ __host__ DeviceVector() : count{ 0 } {  }
	__device__ __host__ size_t size()
	{
		return count;
	}

	__device__ __host__ void push_back(T element)
	{
		if (count < 32)
		{
			array[count++] = element;
		}
	}
	__device__ __host__ T operator[](size_t index)
	{
		return array[index];
	}
};