#pragma once
#include <iostream>
#include <fstream>
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

#define BLOCKS 32
#define THREADS 768
#define ITERATIONS 128

#define HUMANPLAYER 'h'
#define GPUPLAYER 'g'
#define CPUPLAYER 'c'


//#include "TreeCuda.cuh"

void runGameWindow(Window& w, std::condition_variable& cv);
void runGame();


int main(int argc, char** argv)
{
	srand(time(NULL));

	std::cout << "Show window (y/n): ";
	char in;
	std::cin >> in;
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

	const unsigned long long seed = 1232;
	curandState* devStates;
	cudaMalloc((void**)&devStates, THREADS * BLOCKS * sizeof(curandState));
	initializeRandomStates << <BLOCKS, THREADS >> > (devStates, seed);

	Board board = createBoard();

	float cWhite = 2.0f, cBlack = 2.0f;
	std::cout << "wspolczynnik c dla bialego: ";
	std::cin >> cWhite;

	std::cout << "wspolczynnik c dla czarnego: ";
	std::cin >> cBlack;

	bool winner, hasWinner;
	while (generateMoves(board).size() > 0 && !isTerminal(board, &hasWinner, &winner))
	{
		float c = board.currentTurn == WHITE ? cWhite : cBlack;
		Tree tree{ board, c };

		for (int count = 0; count < ITERATIONS; count++)
		{
			TreeNode* p = tree.select();
			auto nodes = tree.expand(p);
			if (nodes.size() == 0)
			{
				continue;
			}

			int* whiteWins = new int[nodes.size()];
			int* blackWins = new int[nodes.size()];
			int* totalPlayed = new int[nodes.size()];
			std::memset(whiteWins,   0,sizeof(int) * nodes.size());
			std::memset(blackWins,   0,sizeof(int) * nodes.size());
			std::memset(totalPlayed, 0,sizeof(int) * nodes.size());

			tree.playoutCuda(nodes, whiteWins, blackWins, totalPlayed, devStates, THREADS, BLOCKS);

			for (int i = 0; i < nodes.size(); i++)
			{
				TreeNode* pp = nodes[i];
				tree.backpropagation(pp, whiteWins[i], blackWins[i], totalPlayed[i]);
			}
			delete whiteWins;
			delete blackWins;
			delete totalPlayed;

		}
		board = tree.getBestMove();
		std::cout << board << std::endl;
	}
	cudaFree(devStates);
}

void runGameWindow(Window& w, std::condition_variable& cv)
{
	char whiteMode, blackMode;
	std::cout << "White human/Gpu/Cpu: (h/g/c): ";
	std::cin >> whiteMode;
	std::cout << "Black human/Gpu/Cpu: (h/g/c): ";
	std::cin >> blackMode;

	const unsigned long long seed = 1234;
	
	curandState* devStates;
	cudaMalloc((void**)&devStates, THREADS * BLOCKS * sizeof(curandState));
	initializeRandomStates << <BLOCKS, THREADS >> > (devStates, seed);
	
	
	Board board = createBoard();
	w.updateBoard(board);

	std::mutex m;
	while(generateMoves(board).size() > 0)
	{
		if ((board.currentTurn == WHITE && whiteMode == HUMANPLAYER) || (board.currentTurn == BLACK && blackMode == HUMANPLAYER))
		{
			std::unique_lock<std::mutex> lock(m);
			cv.wait(lock, [&]() {
				return !(isTheSameBoard(board, w.boardUpdated()));
				}
			);
			board = w.boardUpdated();
			lock.unlock();
			continue;
		}
		Tree tree{ board };

		std::chrono::milliseconds counting{};


		//while (counting < maxTimeMove)
		for(int i = 0; i < ITERATIONS; i++)
		{
			TreeNode* p = tree.select();
			bool hasWinner, winner;
			if (isTerminal(p->position, &hasWinner, &winner) && hasWinner)
			{
				tree.backpropagation(p, (winner == WHITE && hasWinner) ? THREADS : 0, (winner == WHITE && hasWinner) ? 0 : THREADS, THREADS);
				continue;
			}
			auto nodes = tree.expand(p);
			auto start = std::chrono::high_resolution_clock::now();

			int* whiteWins = new int[nodes.size()];
			int* blackWins = new int[nodes.size()];
			int* totalPlayed = new int[nodes.size()];
			std::memset(whiteWins,   0,sizeof(int) * nodes.size());
			std::memset(blackWins,   0,sizeof(int) * nodes.size());
			std::memset(totalPlayed, 0,sizeof(int) * nodes.size());

			if ((board.currentTurn == WHITE && whiteMode == GPUPLAYER) || (board.currentTurn == BLACK && blackMode == GPUPLAYER))
			{
				tree.playoutCuda(nodes, whiteWins, blackWins, totalPlayed, devStates, THREADS, BLOCKS);
			}
			else if ((board.currentTurn == WHITE && whiteMode == CPUPLAYER) || (board.currentTurn == BLACK && blackMode == CPUPLAYER))
			{
				tree.playout(nodes, whiteWins, blackWins, totalPlayed);	
			}
			
			for (int j = 0; j < nodes.size(); j++)
			{
				TreeNode* pp = nodes[j];
				tree.backpropagation(pp, whiteWins[j], blackWins[j], totalPlayed[j]);
			}
			delete whiteWins;
			delete blackWins;
			delete totalPlayed;

			auto end = std::chrono::high_resolution_clock::now();
			counting += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		}
		board = tree.getBestMove();
		w.updateBoard(board);
	}
	std::cout << "Finished \n";
	cudaFree(devStates);
	w.finishGame();

}

