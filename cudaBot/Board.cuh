#pragma once

#include <cuda_runtime.h>
#include "DeviceVector.cuh"
#include <iostream>

#define BLACK false
#define	WHITE true
#define MAXMOVESNOBEATS 30
#define SHIFTDOWN(mask, amount) (mask & additionalShiftDownMask) > 0 ? (mask << (amount + 1)) : (mask << amount);
#define SHIFTUP(mask, amount) (mask & additionalShiftDownMask) == 0 ? (mask >> (amount + 1)) : (mask >> amount);


const uint32_t lowerBorder = 0xf0000000;
const uint32_t upperBorder = 0xf;
const uint32_t rightBorder = 0x80808080;
const uint32_t leftBorder = 0x01010101;

const uint32_t additionalShiftDownMask = 0xf0f0f0f0;

struct Board
{
	uint32_t whitePawns;
	uint32_t whiteQueens;
	uint32_t blackPawns;
	uint32_t blackQueens;
	bool currentTurn;
	uint8_t lastBeat;
};
__device__ __host__ Board createBoard();
__device__ __host__ Board createBoard(uint32_t whitePawns, uint32_t blackPawns, uint32_t whiteQueens, uint32_t blackQueens, bool currentTurn, uint8_t lastBeat);
__device__ __host__ void generatePawnBeatMoves(uint32_t mask, DeviceVector<uint32_t>& positions, Board b);
__device__ __host__ bool isTerminal(Board board, bool* hasWinner, bool* winner);
__device__ __host__ void generateQueenMoves(uint32_t mask, DeviceVector<uint32_t>& positions, Board b);
__device__ __host__ void generateFromFigure(uint32_t mask, DeviceVector<uint32_t>& positions, Board b);
__device__ __host__ void generatePawnBeatMoves(uint32_t mask, DeviceVector<uint32_t>& positions, Board b);
__device__ __host__ DeviceVector<uint32_t> generateMoves(Board b);
__host__ bool isTheSameBoard(Board b1, Board b2);
__device__ __host__ Board applyMove(Board board, uint32_t move);
//__device__ __host__ uint32_t shift(uint32_t mask, int amount);

std::ostream& operator<<(std::ostream& out, Board& board);


//class BoardUtils
//{
//public:
//	__device__ __host__ Board CreateBoard();
//	__device__ __host__ Board CreateBoard(uint64_t whitePawns, uint64_t blackPawns, uint64_t whiteQueens, uint64_t blackQueens, bool currentTurn);
//	__device__ __host__ DeviceVector<Board> generatePositions();
//	__device__ __host__ bool isTerminal(bool* winner);
//	__device__ __host__ Board& operator=(const Board& other);
//	__device__ __host__ DeviceVector<Board> generateFromFigure(uint64_t mask, DeviceVector<Board>& positions);
//	__device__ __host__ bool operator==(const Board& other);
//	friend std::ostream& operator<<(std::ostream& out, Board& board);
//
//	const uint64_t lowerBorder = 0xff00000000000000;
//	const uint64_t upperBorder = 0xff;
//	const uint64_t rightBorder = 0x8080808080808080;
//	const uint64_t leftBorder = 0x0101010101010101;
//	const uint8_t maxPositions = 32;
//	bool currentTurn;
//
//	uint64_t whitePawns;
//	uint64_t whiteQueens;
//	uint64_t blackPawns;
//	uint64_t blackQueens;
//private:
//	__device__ __host__ void generatePawnBeatMoves(uint64_t mask, DeviceVector<Board>& positions);
//	__device__ __host__ void promoteToQueen(uint64_t& pawns, uint64_t& queens, uint64_t mask);
//
//	__device__ __host__ void generateQueenMoves(uint64_t mask, DeviceVector<Board>& positions);
//
//};





