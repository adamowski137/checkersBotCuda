#pragma once
#include <iostream>
#include <bitset>
#include <thread>
#include <chrono>
#include "Tree.cuh"
#include "Board.cuh"
#include <algorithm>
#include "Window.hpp"
#include <ctime>
#include <chrono>
#include <condition_variable>
#include <mutex>

#define BLOCKS 100
#define THREADS 256

//#include "TreeCuda.cuh"

void runGameWindow(Window& w, std::condition_variable& cv);
void runGame();


int main(int argc, char** argv)
{
	std::cout << "Show window (y/n): ";
	srand(time(NULL));
	char in;
	//std::cin >> in;
	in = 'n';
	if (in == 'y')
	{
		std::condition_variable cv{};
		Window w{cv};
		std::thread t(runGameWindow, std::ref(w), std::ref(cv));
		w.runWindow();
		t.join();
	}
	if (in == 'n')
	{
		std::thread t(runGame);
		t.join();
	}




	return 0;
}

void runGame()
{
	std::chrono::milliseconds maxTimeMove(1000);

	const unsigned long long seed = 1232;
	curandState* devStates;
	cudaMalloc((void**)&devStates, THREADS * BLOCKS * sizeof(curandState));
	initializeRandomStates << <BLOCKS, THREADS >> > (devStates, seed);


	//Board board{64, 32768, 0, 0, WHITE};
	Board board{};

	std::cout << board << std::endl;

	int count = 0;
	while (board.generatePositions().size() > 0 && count < 10)
	{
		count++;
		float c = board.currentTurn == WHITE ? 2.5f : 2.0f;
		Tree tree{ board, c };

		std::chrono::milliseconds counting{};


		//while (counting < maxTimeMove)
		for (int i = 0; i < 16; i++)
		{
			TreeNode* p = tree.select();
			auto nodes = tree.expand(p);
			auto start = std::chrono::high_resolution_clock::now();
			if (nodes.size() == 0)
			{
				continue;
			}

			int* whiteWins = new int[nodes.size()];
			int* blackWins = new int[nodes.size()];
			int* totalPlayed = new int[nodes.size()];
			try
			{
				if (board.currentTurn == WHITE)
				{
					tree.playoutCuda(nodes, whiteWins, blackWins, totalPlayed, devStates, THREADS);
					for (int i = 0; i < nodes.size(); i++)
					{
						tree.backpropagation(nodes[i], whiteWins[i], blackWins[i], totalPlayed[i]);
					}
				}
				else
				{
					/*int idx = rand() % nodes.size();
					tree.playout(nodes[idx], &whiteWins[idx], &blackWins[idx], &totalPlayed[idx]);
					tree.backpropagation(nodes[idx], whiteWins[idx], blackWins[idx], totalPlayed[idx]);*/
					tree.playoutCuda(nodes, whiteWins, blackWins, totalPlayed, devStates, THREADS);
					for (int i = 0; i < nodes.size(); i++)
					{
						tree.backpropagation(nodes[i], whiteWins[i], blackWins[i], totalPlayed[i]);
					}
				
				}
			}
			catch (std::string s)
			{
				std::cout << s;
				return;
			}
			delete whiteWins;
			delete blackWins;
			delete totalPlayed;

			auto end = std::chrono::high_resolution_clock::now();
			counting += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		}
		board = tree.getBestMove();
		std::cout << board << std::endl;
	}
	std::cout << "Finished \n";
	cudaFree(devStates);
}

void runGameWindow(Window& w, std::condition_variable& cv)
{
	std::chrono::milliseconds maxTimeMove(1000);

	const unsigned long long seed = 1234;
	curandState* devStates;
	cudaMalloc((void**)&devStates, THREADS * BLOCKS * sizeof(curandState));
	initializeRandomStates << <BLOCKS, THREADS >> > (devStates, seed);


	//Board board{64, 32768, 0, 0, WHITE};
	Board board{};

	bool whitePlayer = true;
	bool blackPlayer = false;

	std::cout << board << std::endl;
	w.updateBoard(board);

	std::mutex m;
	int count = 0;
	while(board.generatePositions().size() > 0)
	{
		count++;
		if ((board.currentTurn == WHITE && whitePlayer) || (board.currentTurn == BLACK && blackPlayer))
		{
			std::unique_lock<std::mutex> lock(m);
			cv.wait(lock, [&]() {
				return !(board == w.boardUpdated());
				}
			);
			board = w.boardUpdated();
			lock.unlock();
			continue;
		}
		Tree tree{ board };

		std::chrono::milliseconds counting{};


		//while (counting < maxTimeMove)
		for(int i = 0; i < 10; i++)
		{
			TreeNode* p = tree.select();
			auto nodes = tree.expand(p);
			auto start = std::chrono::high_resolution_clock::now();
			if (nodes.size() == 0)
			{
				continue;
			}

			int* whiteWins = new int[nodes.size()];
			int* blackWins = new int[nodes.size()];
			int* totalPlayed = new int[nodes.size()];
			try 
			{
				tree.playoutCuda(nodes, whiteWins, blackWins, totalPlayed, devStates, THREADS);
			}
			catch(std::string s)
			{
				std::cout << s;
				return;
			}
			for (int i = 0; i < nodes.size(); i++)
			{
				tree.backpropagation(nodes[i], whiteWins[i], blackWins[i], totalPlayed[i]);
			}
			delete whiteWins;
			delete blackWins;
			delete totalPlayed;

			auto end = std::chrono::high_resolution_clock::now();
			counting += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		}
		board = tree.getBestMove();
		w.updateBoard(board);
		std::cout << board << std::endl;
	}
	std::cout << "Finished \n";
	cudaFree(devStates);
	w.finishGame();

}

