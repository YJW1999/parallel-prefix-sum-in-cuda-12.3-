#pragma once

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define size 500

__device__ int ScanBlock_HillisSteele(int val, int* smem)
{
	smem[threadIdx.x] = val;
	__syncthreads();

	for (int offset = 1; offset < blockDim.x; offset <<= 1)
	{
		if (threadIdx.x >= offset) val += smem[threadIdx.x-offset];
		__syncthreads();

		smem[threadIdx.x] = val;
		__syncthreads();
	}

	return smem[threadIdx.x];
}

__device__ int DeviceScan(int val, int* smem, int* sum, int* counter) {
	val = ScanBlock_HillisSteele(val, smem);
	__shared__ int offset;
	if (threadIdx.x == blockDim.x - 1) {
		while (atomicAdd(counter, 0) < blockIdx.x);
			
		offset = atomicAdd(sum, val);
		// Use memory fence to ensure the order of atomics
		__threadfence();
		// Signalize the next block that can be processed
		atomicAdd(counter, 1);
	}

	__syncthreads();

	return offset + val;
}

__global__ void parallelPrefixKernel(const int* input, int* output, int _size, int* sum, int* counter) {
	extern __shared__ int smem[];
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + tid;

	int val = 0;
	if (gid < _size) val = input[gid];

	val = DeviceScan(val, smem, sum, counter);

	if (gid < _size) {
		output[gid] = val;
	}
}
 
int main() {
	int* data = new int[size];
	for (int i = 0; i < size; ++i) {
		data[i] = rand() % 1000;
	}

	int* input_data, * output_data;
	cudaMalloc((void**)&input_data, size * sizeof(int));
	cudaMalloc((void**)&output_data, size * sizeof(int));
	cudaMemcpy(input_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

	int* host_sum = 0, * host_counter = 0;

	int* sum, *counter;
	cudaMalloc((void**)&sum, sizeof(int));
	cudaMalloc((void**)&counter, sizeof(int));
	cudaMemcpy(sum, host_sum, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(counter, host_counter, sizeof(int), cudaMemcpyHostToDevice);
	
	//set configuration for block size and number of blocks
	const int blockSize = 256;
	const int gridSize = (size + blockSize - 1) / blockSize;

	//cuda Kernel function call
	parallelPrefixKernel <<<gridSize, blockSize >>> (input_data, output_data, size, sum, counter);

	int* output = new int[size];
	cudaMemcpy(output, output_data, size * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; ++i) {
		std::cout << data[i] << ": " << output[i] << std::endl;
	}

	return 0;
}
