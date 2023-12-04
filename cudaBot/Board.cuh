#pragma once

#include <cuda_runtime.h>
#include "DeviceVector.cuh"
#include <iostream>

#define BLACK false
#define	WHITE true

class Board
{
public:
	__device__ __host__ Board();
	__device__ __host__ Board(uint64_t whitePawns, uint64_t blackPawns, uint64_t whiteQueens, uint64_t blackQueens, bool currentTurn);
	__device__ __host__ DeviceVector<Board> generatePositions();
	__device__ __host__ bool isTerminal(bool* winner);
	__device__ __host__ Board& operator=(const Board& other);
	__device__ __host__ DeviceVector<Board> generateFromFigure(uint64_t mask, DeviceVector<Board>& positions);
	__device__ __host__ bool operator==(const Board& other);
	friend std::ostream& operator<<(std::ostream& out, Board& board);

	const uint64_t lowerBorder = 0xff00000000000000;
	const uint64_t upperBorder = 0xff;
	const uint64_t rightBorder = 0x8080808080808080;
	const uint64_t leftBorder = 0x0101010101010101;
	const uint8_t maxPositions = 32;
	bool currentTurn;

	uint64_t whitePawns;
	uint64_t whiteQueens;
	uint64_t blackPawns;
	uint64_t blackQueens;
private:
	__device__ __host__ void generatePawnBeatMoves(uint64_t mask, DeviceVector<Board>& positions);
	__device__ __host__ void promoteToQueen(uint64_t& pawns, uint64_t& queens, uint64_t mask);

	__device__ __host__ void generateQueenMoves(uint64_t mask, DeviceVector<Board>& positions);

};
