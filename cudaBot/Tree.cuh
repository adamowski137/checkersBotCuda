#pragma once
#include "Board.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include "DeviceVector.cuh"
#include <iostream>
#include <curand_kernel.h>

__global__ void initializeRandomStates(curandState* state, unsigned long long seed)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__global__ void threadWork(DeviceVector<Board> boards, int* whiteWins, int* blackWins, int* totalPlayed, curandState* state)
{
	Board position = boards[blockIdx.x];
	bool winner;
	int i = 0;
	atomicAdd(&totalPlayed[blockIdx.x], 1);
	// average game length is ~49 moves
	while (!position.isTerminal(&winner) && i < 100)
	{
		//DeviceVector<Board> positions{};
		DeviceVector<Board> positions = position.generatePositions();
		if (positions.size() == 0)
		{
			return;
		}
		curandState localState = state[blockIdx.y * blockIdx.x * 256 + threadIdx.x];
		int randomIndex = curand(&localState) % positions.size();
		state[threadIdx.x] = localState;
		position = positions[randomIndex];
		i++;
	}
	if (winner == WHITE)
	{
		atomicAdd(&whiteWins[blockIdx.x], 1);

		//*whiteWins += 1;
	}
	if (winner == BLACK)
	{
		atomicAdd(&blackWins[blockIdx.x], 1);

		//*blackWins += 1;
	}
}

struct TreeNode
{
	Board position;
	TreeNode* prev;
	DeviceVector<TreeNode*> next;
	int blackVictories;
	int whiteVictories;
	int totalPlayed;
};

class Tree
{

public:

	TreeNode* root;


	__host__ Tree(Board position, float c = 2) : c{c}
	{
		root = new TreeNode;
		root->position = position;
		root->blackVictories = 0;
		root->whiteVictories = 0;
		root->totalPlayed = 0;
		root->next = DeviceVector<TreeNode*>{};
		root->prev = NULL;
	}

	__host__ ~Tree()
	{
		if (root != NULL)
		{
			freeNode(root);
		}
	}

	__host__ void freeNode(TreeNode* p)
	{
		for (int i = 0; i < p->next.size(); i++)
		{
			freeNode(p->next[i]);
		}
		delete p;
	}


	__host__ bool isLeaf(TreeNode* p)
	{
		return p->next.size() == 0;
	}
	__host__ TreeNode* selectNode(TreeNode* p)
	{
		int maxi = 0;
		float maxValue = 0.0f;
		int N = p->totalPlayed;
		for (int i = 0; i < p->next.size(); i++)
		{
			int n = p->totalPlayed;
			float c = 2.0f;
			if (p->position.currentTurn == WHITE)
			{

				float value = (p->next[i]->whiteVictories / (float)(n)) + sqrtf(c * logf((float)N) / n);
				if (maxValue < value)
				{
					maxValue = value;
					maxi = i;
				}
			}
			if (p->position.currentTurn == BLACK)
			{
				float value = (p->next[i]->blackVictories / (float)(n)) + sqrtf(c * logf((float)N) / n);
				if (maxValue < value)
				{
					maxValue = value;
					maxi = i;
				}
			}
		}
		return p->next[maxi];
	}

	__host__ TreeNode* select()
	{
		TreeNode* p = root;
		while (!isLeaf(p))
		{
			p = selectNode(p);
		}
		return p;
	}

	__host__ DeviceVector<TreeNode*> expand(TreeNode* p)
	{
		DeviceVector<Board> postions = p->position.generatePositions();
		for (int i = 0; i < postions.size(); i++)
		{
			TreeNode* t = new TreeNode;
			t->position = postions[i];
			t->whiteVictories = 0;
			t->blackVictories = 0;
			t->totalPlayed = 0;
			t->next = DeviceVector<TreeNode*>{};
			t->prev = p;

			p->next.push_back(t);
		}
		return p->next;
	}

	__host__ int playout(TreeNode* p, int* whiteWins, int* blackWins, int* totalPlayed)
	{
		*totalPlayed += 1;
		Board position = p->position;
		bool winner;
		int i = 0;
		while (!position.isTerminal(&winner) && i < 100)
		{
			auto positions = position.generatePositions();
			if (positions.size() == 0)
			{
				return 0;
			}
			position = positions[rand() % positions.size()];
			i++;
		}
		if (i == 100)
		{
			return 0;
		}
		if (winner == WHITE)
		{
			*whiteWins += 1;
		}
		else if (winner == BLACK)
		{
			*blackWins += 1;
		}

		return winner == WHITE ? 1 : -1;
	}

	__host__ int playoutCuda(DeviceVector<TreeNode*>& nodes, int* whiteWins, int* blackWins, int* totalPlayed, curandState* devStates, int threads)
	{
		DeviceVector<Board> boards;
		for (int i = 0; i < nodes.size(); i++) 
		{
			boards.push_back(nodes[i]->position);
		}

		cudaError_t cudaStatus;
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		int* dev_whiteWins, * dev_blackWins, * dev_totalPlayed;

		cudaMalloc((void**)&dev_whiteWins, boards.size() * sizeof(int));
		cudaMalloc((void**)&dev_blackWins, boards.size() * sizeof(int));
		cudaMalloc((void**)&dev_totalPlayed, boards.size() * sizeof(int));

		cudaMemset(dev_whiteWins, 0, boards.size() * sizeof(int));
		cudaMemset(dev_blackWins, 0, boards.size() * sizeof(int));
		cudaMemset(dev_totalPlayed, 0, boards.size() * sizeof(int));

		dim3 blocks;
		blocks.x = boards.size();
		blocks.y = 10;

		threadWork << <blocks, threads>> > (boards, dev_whiteWins, dev_blackWins, dev_totalPlayed, devStates);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			throw(cudaGetErrorString(cudaStatus));
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			throw("cudaDeviceSynchronize returned error code %d after launching addKernel!\n");
		}

		cudaStatus = cudaMemcpy(whiteWins, dev_whiteWins, boards.size() * sizeof(int), cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(blackWins, dev_blackWins, boards.size() * sizeof(int), cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(totalPlayed, dev_totalPlayed,boards.size() *  sizeof(int), cudaMemcpyDeviceToHost);


		cudaFree(dev_whiteWins);
		cudaFree(dev_blackWins);
		cudaFree(dev_totalPlayed);
		return 1;
	}

	__host__ void backpropagation(TreeNode* p, int whiteWins, int blackWins, int totalPlayed)
	{
		while (p != NULL)
		{
			p->whiteVictories += whiteWins;
			p->blackVictories += blackWins;
			p->totalPlayed += totalPlayed;
			p = p->prev;
		}
	}

	__host__ Board getBestMove()
	{
		int max = 0;
		int maxi = 0;
		for (int i = 0; i < root->next.size(); i++) {
			
			bool winner;
			if (root->next[i]->position.isTerminal(&winner))
			{
				return root->next[i]->position;
			}
			int n = root->next[i]->totalPlayed;
			if (max < n)
			{
				max = n;
				maxi = i;
			}
		}
		std::cout << (root->next[maxi]->totalPlayed) << std::endl;
		return root->next[maxi]->position;
	}
private:
	float c;

};

