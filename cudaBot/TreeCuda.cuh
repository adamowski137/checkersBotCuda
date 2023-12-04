#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Board.cuh"

__global__ void threadWork(LocalVector<Board> positions, float* white, float* black, bool color, int blocks, int threads)
{

	/*Board position = positions[blockIdx.x];
	while (true)
	{
		bool winner;
		bool terminal = false;
		if ((position.whitePawns | position.whiteQueens) == 0)
		{
			winner = BLACK;
			terminal = true;
		}

		if ((position.blackPawns | position.blackQueens) == 0)
		{
			winner = WHITE;
			terminal = true;
		}

		if (terminal)
		{
			if (winner == WHITE)
			{
				white[blockIdx.x] += 1.0f;
				return;
			}
			else
			{
				black[blockIdx.x] += 1.0f;
				return;
			}
		}
		return;
	}*/
}

Board getBestMove(Board position, bool color)
{
	cudaError_t cudaStatus;

	auto positions = position.generatePositions(color);
	int blocks = positions.size();
	int threads = 1024;

	float* black = new float[blocks];
	float* white = new float[blocks];

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	float* dev_black;
	float* dev_white;

	cudaMalloc((void**)&dev_black, sizeof(float) * blocks);
	cudaMalloc((void**)&dev_white, sizeof(float) * blocks);

	cudaMemcpy(dev_black, black, sizeof(float) * blocks, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_white, white, sizeof(float) * blocks, cudaMemcpyHostToDevice);

	threadWork<<<blocks, threads>>>(positions, dev_white, dev_black, color, blocks, threads);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}


	cudaMemcpy(dev_black, black, sizeof(float) * blocks, cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_white, white, sizeof(float) * blocks, cudaMemcpyDeviceToHost);

	cudaFree(dev_white);
	cudaFree(dev_black);

	float bestWhiteRatio = white[0] / (white[0] + black[0]);
	float bestBlackRatio = black[0] / (white[0] + black[0]);
	int blackIdx = 0;
	int whiteIdx = 0;

	for (int i = 0; i < blocks; i++)
	{
		if (bestWhiteRatio < white[i] / (white[i] + black[i]))
		{
			whiteIdx = i;
			bestWhiteRatio = white[i] / (white[i] + black[i]);
		}
		if (bestBlackRatio < black[i] / (white[i] + black[i]))
		{
			blackIdx = i;
			bestBlackRatio = black[i] / (white[i] + black[i]);
		}
	}

	delete white;
	delete black;

	return (color == WHITE ? positions[whiteIdx] : positions[blackIdx]);
Error:

	return Board();
}
