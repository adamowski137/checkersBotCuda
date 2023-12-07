#pragma once
#include "Board.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include "DeviceVector.cuh"
#include <fstream>
#include <curand_kernel.h>



__global__ void initializeRandomStates(curandState* state, unsigned long long seed)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__global__ void threadWork(Board* boards, int* whiteWins, int* blackWins, int* totalPlayed, curandState* state, int moves)
{
	int moveIdx = blockIdx.x;
	Board position = boards[moveIdx];
	bool winner, hasWinner;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	//totalPlayed[blockIdx.x] +=  1;
	// average game length is ~49 moves
	while (!isTerminal(position, &hasWinner, &winner) )
	{
		//DeviceVector<Board> positions{};
		DeviceVector<uint32_t> moves = generateMoves(position);
		if (moves.size() == 0)
		{
			return;
		}
		int randomIndex = (curand_uniform(state + idx) * (moves.size()));
		if (randomIndex == moves.size())	randomIndex--;
		position = applyMove(position, moves[randomIndex]);
	}
	if (!hasWinner)
	{
		return;
	}

	if (winner == WHITE)
	{
		atomicAdd(&whiteWins[moveIdx], 1);
		atomicAdd(&totalPlayed[moveIdx], 1);
		return;
		//*whiteWins += 1;
	}
	if (winner == BLACK)
	{
		atomicAdd(&totalPlayed[moveIdx], 1);
		atomicAdd(&blackWins[moveIdx], 1);
		return;

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
	std::ofstream out;

	__host__ Tree(Board position, float c = 2) : c{c}
	{
		root = new TreeNode;
		root->position = position;
		root->blackVictories = 0;
		root->whiteVictories = 0;
		root->totalPlayed = 0;
		root->next = DeviceVector<TreeNode*>{};
		root->prev = NULL;
		out = std::ofstream{ "times.txt" };
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
			int n = p->next[i]->totalPlayed;
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
		auto startSelect = std::chrono::high_resolution_clock::now();
		TreeNode* p = root;
		while (!isLeaf(p))
		{
			p = selectNode(p);
		}
		auto endSelect = std::chrono::high_resolution_clock::now();
		out << "select: " << std::chrono::duration_cast<std::chrono::microseconds>(endSelect - startSelect).count() << " microseconds" << std::endl;
		return p;
	}

	__host__ DeviceVector<TreeNode*> expand(TreeNode* p)
	{
		DeviceVector<uint32_t> moves = generateMoves(p->position);

		for (int i = 0; i < moves.size(); i++)
		{
			TreeNode* t = new TreeNode;
			t->position = applyMove(p->position, moves[i]);
			t->whiteVictories = 0;
			t->blackVictories = 0;
			t->totalPlayed = 0;
			t->next = DeviceVector<TreeNode*>{};
			t->prev = p;

			p->next.push_back(t);
		}
		return p->next;
	}

	__host__ void playout(DeviceVector<TreeNode*>& nodes, int* whiteWins, int* blackWins, int* totalPlayed)
	{
		for (int i = 0; i < nodes.size(); i++)
		{
			Board position = nodes[i]->position;
			bool winner, hasWinner;
			while (isTerminal(position, &hasWinner, &winner))
			{
				auto positions = generateMoves(position);
				if (positions.size() == 0)
				{
					break;
				}
				position = applyMove(position, positions[rand() % positions.size()]);
			}
			if (!hasWinner)
			{
				return;
			}
			if (winner == WHITE)
			{
				totalPlayed[i] += 1;
				whiteWins[i] += 1;
			}
			else if (winner == BLACK)
			{
				totalPlayed[i] += 1;
				blackWins[i] += 1;
			}
		}

	}

	__host__ int playoutCuda(DeviceVector<TreeNode*>& nodes, int* whiteWins, int* blackWins, int* totalPlayed, curandState* devStates, int threads, int blocks)
	{
		int* tmpWhiteWins, *tmpBlackWins, *tmpTotalPlayed;
		tmpWhiteWins = new int[blocks];
		tmpBlackWins = new int[blocks];
		tmpTotalPlayed = new int[blocks];

		std::memset(tmpWhiteWins, 0, blocks * sizeof(int));
		std::memset(tmpBlackWins, 0, blocks * sizeof(int));
		std::memset(tmpTotalPlayed, 0, blocks * sizeof(int));
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
		Board* dev_boards;
		auto startMalloc = std::chrono::high_resolution_clock::now();
		cudaMalloc((void**)&dev_whiteWins, blocks * sizeof(int));
		cudaMalloc((void**)&dev_blackWins, blocks * sizeof(int));
		cudaMalloc((void**)&dev_totalPlayed, blocks * sizeof(int));
		cudaMalloc((void**)&dev_boards, blocks * sizeof(Board));


		cudaMemset(dev_whiteWins, 0, blocks * sizeof(int));
		cudaMemset(dev_blackWins, 0, blocks * sizeof(int));
		cudaMemset(dev_totalPlayed, 0, blocks * sizeof(int));
		cudaMemcpy(dev_boards, boards.array, boards.size() * sizeof(Board), cudaMemcpyHostToDevice);
		auto endMalloc = std::chrono::high_resolution_clock::now();
		out << "cuda malloc + memcpy: " << std::chrono::duration_cast<std::chrono::microseconds>(endMalloc - startMalloc).count() << " microseconds" << std::endl;


		auto startExec = std::chrono::high_resolution_clock::now();
		threadWork << <blocks, threads >> > (dev_boards, dev_whiteWins, dev_blackWins, dev_totalPlayed, devStates, boards.size());


		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			throw(cudaGetErrorString(cudaStatus));
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			throw("cudaDeviceSynchronize returned error code %d after launching addKernel!\n");
		}
		auto endExec = std::chrono::high_resolution_clock::now();
		out << "cuda exec: " << std::chrono::duration_cast<std::chrono::microseconds>(endExec - startExec).count() << " microseconds" << std::endl;
		auto startFree = std::chrono::high_resolution_clock::now();

		cudaStatus = cudaMemcpy(tmpWhiteWins, dev_whiteWins, blocks * sizeof(int), cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(tmpBlackWins, dev_blackWins, blocks * sizeof(int), cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(tmpTotalPlayed, dev_totalPlayed, blocks *  sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(dev_whiteWins);
		cudaFree(dev_blackWins);
		cudaFree(dev_totalPlayed);
		cudaFree(dev_boards);

		auto endFree = std::chrono::high_resolution_clock::now();
		out << "cuda cpy + free: " << std::chrono::duration_cast<std::chrono::microseconds>(endFree - startFree).count() << " microseconds" << std::endl;

		for (int i = 0; i < blocks; i++)
		{
			int idx = i % boards.size();
			whiteWins[idx] += tmpWhiteWins[i];
			blackWins[idx] += tmpBlackWins[i];
			totalPlayed[idx] += tmpTotalPlayed[i];
		}

		delete tmpWhiteWins;
		delete tmpBlackWins;
		delete tmpTotalPlayed;

		return 1;
	}

	__host__ void backpropagation(TreeNode* last, int whiteWins, int blackWins, int totalPlayed)
	{
		auto startBackpropagation = std::chrono::high_resolution_clock::now();

		TreeNode* p = last;
		while (p != NULL)
		{
			(p)->whiteVictories += whiteWins;
			(p)->blackVictories += blackWins;
			(p)->totalPlayed += totalPlayed;
			p = ((p)->prev);
		}

		auto endBackpropagation = std::chrono::high_resolution_clock::now();
		out << "backpropagation: " << std::chrono::duration_cast<std::chrono::microseconds>(endBackpropagation - startBackpropagation).count() << " microseconds" << std::endl;

	}

	__host__ Board getBestMove()
	{
		int max = 0;
		int maxi = 0;
		for (int i = 0; i < root->next.size(); i++) {
			
			int n = root->next[i]->totalPlayed;
			if (max < n)
			{
				max = n;
				maxi = i;
			}
		}
		std::cout << "move found" << std::endl;
		return root->next[maxi]->position;
	}
private:
	float c;

};

