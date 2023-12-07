#pragma once

#include <cuda_runtime.h>
#include "DeviceVector.cuh"
#include <iostream>
#include "Board.cuh"


__device__ __host__ Board createBoard()
{
	Board b;
	b.whitePawns = 0xfff;
	b.blackPawns = 0xfff00000;
	b.whiteQueens = 0;
	b.blackQueens = 0;
	b.currentTurn = WHITE;
	b.lastBeat = 0;
	return b;
}

__device__ __host__ Board createBoard(uint32_t whitePawns, uint32_t blackPawns, uint32_t whiteQueens, uint32_t blackQueens, bool currentTurn, uint8_t lastBeat)
{
	Board b;
	b.whitePawns = whitePawns;
	b.blackPawns = blackPawns;
	b.whiteQueens = whiteQueens;
	b.blackQueens = blackQueens;
	b.currentTurn = currentTurn;
	b.lastBeat = lastBeat;
	return b;
}

__device__ __host__ bool isTerminal(Board board, bool* hasWinner, bool* winner)
{
	if (board.lastBeat >= MAXMOVESNOBEATS)
	{
		*hasWinner = false;
		return true;
	}
	if ((board.whitePawns | board.whiteQueens) == 0)
	{
		*hasWinner = true;
		*winner = BLACK;
		return true;
	}

	if ((board.blackPawns | board.blackQueens) == 0)
	{
		*hasWinner = true;
		*winner = WHITE;
		return true;
	}
	return false;
}

__device__ __host__ void generateQueenMoves(uint32_t mask, DeviceVector<uint32_t>& positions, Board b)
{
	uint32_t whitePawns = b.whitePawns;
	uint32_t blackPawns = b.blackPawns;
	uint32_t whiteQueens = b.whiteQueens;
	uint32_t blackQueens = b.blackQueens;
	bool currentTurn = b.currentTurn;

	uint32_t allFigures = (whitePawns | blackPawns | whiteQueens | blackQueens);
	uint32_t whiteFigures = (whitePawns | whiteQueens);
	uint32_t blackFigures = (blackPawns | blackQueens);


	if (currentTurn == WHITE)
	{
		if ((mask & leftBorder) == 0)
		{
			uint32_t leftUp = SHIFTUP(mask, 4);
			uint32_t current = mask;

			while (leftUp > 0 && (leftUp & whiteFigures) == 0 && (current & leftBorder) == 0)
			{
				if ((leftUp & allFigures) > 0)
				{
					if ((leftUp & blackFigures) > 0 && (leftUp & leftBorder) == 0)
					{
						uint32_t leftBeat = SHIFTUP(leftUp, 4);
						if (leftBeat > 0 && (leftBeat & allFigures) == 0)
						{
							positions.push_back((mask | leftUp | leftBeat));
						}
					}
					break;
				}

				positions.push_back((mask | leftUp));
				current = leftUp;
				leftUp = SHIFTUP(leftUp, 4);
			}
		}
		if ((mask & rightBorder) == 0)
		{
			uint32_t rightUp = SHIFTUP(mask, 3);
			uint32_t current = mask;

			while (rightUp > 0 && (rightUp & whiteFigures) == 0 && (current & rightBorder) == 0)
			{
				if ((rightUp & allFigures) > 0)
				{
					if ((rightUp & blackFigures) > 0 && (rightUp & rightBorder) == 0)
					{
						uint32_t rightBeat = SHIFTUP(rightUp, 3);
						if (rightBeat > 0 && (rightBeat & allFigures) == 0)
						{
							positions.push_back((mask | rightUp | rightBeat));
						}
					}
					break;
				}

				positions.push_back((mask | rightUp));

				current = rightUp;
				rightUp = SHIFTUP(rightUp, 3);
			}
		}
		if ((mask & leftBorder) == 0)
		{
			uint32_t leftDown = SHIFTDOWN(mask, 3);
			uint32_t current = mask;

			while (leftDown > 0 && (leftDown & whiteFigures) == 0 && (current & leftBorder) == 0)
			{
				if ((leftDown & allFigures) > 0)
				{
					if ((leftDown & blackFigures) > 0 && (leftDown & leftBorder) == 0)
					{
						uint32_t leftDownBeat = SHIFTDOWN(leftDown, 3);
						if (leftDownBeat > 0 && (leftDownBeat & allFigures) == 0)
						{
							positions.push_back((mask | leftDown | leftDownBeat));
						}
					}
					break;
				}

				positions.push_back((mask | leftDown));
				current = leftDown;
				leftDown = SHIFTDOWN(leftDown, 3);
			}
		}

		if ((mask & rightBorder) == 0)
		{
			uint32_t current = mask;
			uint32_t rightDown = SHIFTDOWN(mask, 4);
			while (rightDown > 0 && (rightDown & whiteFigures) == 0 && (current & rightBorder) == 0)
			{
				if ((rightDown & allFigures) > 0)
				{
					if ((rightDown & blackFigures) > 0 && (rightDown & rightBorder) == 0)
					{
						uint32_t rightDownBeat = SHIFTDOWN(rightDown, 4);
						if (rightDownBeat > 0 && (rightDownBeat & allFigures) == 0)
						{
							positions.push_back((mask | rightDown | rightDownBeat));
						}
					}
					break;
				}

				positions.push_back((mask | rightDown));
				current = rightDown;
				rightDown = SHIFTDOWN(rightDown, 4);
			}
		}
	}
	if (currentTurn == BLACK)
	{
		if ((mask & leftBorder) == 0)
		{
			uint32_t leftUp = SHIFTUP(mask, 4);
			uint32_t current = mask;

			while (leftUp > 0 && (leftUp & blackFigures) == 0 && (current & leftBorder) == 0)
			{
				if ((leftUp & allFigures) > 0)
				{
					if ((leftUp & whiteFigures) > 0 && (leftUp & leftBorder) == 0)
					{
						uint32_t leftBeat = SHIFTUP(leftUp, 4);
						if (leftBeat > 0 && (leftBeat & allFigures) == 0)
						{
							positions.push_back((mask | leftUp | leftBeat));
						}
					}
					break;
				}
				
				positions.push_back((mask | leftUp));
				current = leftUp;

				leftUp = SHIFTUP(leftUp, 4);
			}
		}

		if ((mask & rightBorder) == 0)
		{
			uint32_t rightUp = SHIFTUP(mask, 3);
			uint32_t current = mask;

			while (rightUp > 0 && (rightUp & blackFigures) == 0 && (current & rightBorder) == 0)
			{
				if ((rightUp & allFigures) > 0)
				{
					if ((rightUp & whiteFigures) > 0 && (rightUp & rightBorder) == 0)
					{
						uint32_t rightBeat = SHIFTUP(rightUp, 3);
						if (rightBeat > 0 && (rightBeat & allFigures) == 0)
						{
							positions.push_back((mask | rightUp | rightBeat));
						}
					}
					break;
				}

				positions.push_back((mask | rightUp));
				current = rightUp;
				rightUp = SHIFTUP(rightUp, 3);
			}
		}

		if ((mask & leftBorder) == 0)
		{
			uint32_t leftDown = SHIFTDOWN(mask, 3);
			uint32_t current = mask;

			while (leftDown > 0 && (leftDown & blackFigures) == 0 && (current & leftBorder) == 0)
			{
				if ((leftDown & allFigures) > 0)
				{
					if ((leftDown & whiteFigures) > 0 && (leftDown & leftBorder) == 0)
					{
						uint32_t leftDownBeat = SHIFTDOWN(leftDown, 3);
						if (leftDownBeat > 0 && (leftDownBeat & allFigures) == 0)
						{
							positions.push_back((mask | leftDown | leftDownBeat));
						}
					}
					break;
				}

				positions.push_back((mask | leftDown));
				current = leftDown;

				leftDown = SHIFTDOWN(leftDown, 3);
			}
		}

		if ((mask & rightBorder) == 0)
		{
			uint32_t current = mask;
			uint32_t rightDown = SHIFTDOWN(mask, 4);
			while (rightDown > 0 && (rightDown & blackFigures) == 0 && (current & rightBorder) == 0)
			{
				if ((rightDown & allFigures) > 0)
				{

					if ((rightDown & whiteFigures) > 0 && (rightDown & rightBorder) == 0)
					{
						uint32_t rightDownBeat = SHIFTDOWN(rightDown, 4);
						if (rightDownBeat > 0 && (rightDownBeat & allFigures) == 0)
						{
							positions.push_back((mask | rightDown | rightDownBeat));
						}
					}
					break;
				}

				positions.push_back((mask | rightDown));
				current = rightDown;
				rightDown = SHIFTDOWN(rightDown, 4);
			}
		}
	}
}

__device__ __host__ void generateFromFigure(uint32_t mask, DeviceVector<uint32_t>& positions, Board b)
{
	uint32_t whitePawns = b.whitePawns;
	uint32_t blackPawns = b.blackPawns;
	uint32_t whiteQueens = b.whiteQueens;
	uint32_t blackQueens = b.blackQueens;
	bool currentTurn = b.currentTurn;

	uint32_t allFigures = (whitePawns | blackPawns | whiteQueens | blackQueens);
	uint32_t whiteFigures = (whitePawns | whiteQueens);
	uint32_t blackFigures = (blackPawns | blackQueens);

	generatePawnBeatMoves(mask, positions, b);

	if (currentTurn == WHITE)
	{
		if (mask & whiteQueens)
		{
			generateQueenMoves(mask, positions, b);
		}
		if ((mask & whitePawns) > 0)
		{
			uint32_t leftMove = SHIFTDOWN(mask, 3);

			if (leftMove > 0 && (mask & leftBorder) == 0)
			{
				if ((leftMove & allFigures) == 0)
				{
					positions.push_back((mask | leftMove));
				}

			}
			uint32_t rightMove = SHIFTDOWN(mask, 4);

			if (rightMove > 0 && (mask & rightBorder) == 0)
			{
				if ((rightMove & allFigures) == 0)
				{

					positions.push_back((mask | rightMove));
				}
			}
		}
	}
	if (currentTurn == BLACK)
	{
		if (mask & blackQueens)
		{
			generateQueenMoves(mask, positions, b);
		}
		if ((mask & blackPawns) > 0)
		{
			uint32_t leftMove = SHIFTUP(mask, 4);

			if (leftMove > 0 && (mask & leftBorder) == 0)
			{
				if ((leftMove & allFigures) == 0)
				{
					positions.push_back((mask | leftMove));
				}

			}
			uint32_t rightMove = SHIFTUP(mask, 3);

			if (rightMove > 0 && (mask & rightBorder) == 0)
			{
				if ((rightMove & allFigures) == 0)
				{
					positions.push_back((mask | rightMove));
				}
			}
		}
	}
}

__device__ __host__ void generatePawnBeatMoves(uint32_t mask, DeviceVector<uint32_t>& positions, Board b)
{
	uint32_t whitePawns = b.whitePawns;
	uint32_t blackPawns = b.blackPawns;
	uint32_t whiteQueens = b.whiteQueens;
	uint32_t blackQueens = b.blackQueens;
	bool currentTurn = b.currentTurn;
	uint32_t allFigures = (whitePawns | blackPawns | whiteQueens | blackQueens);
	uint32_t whiteFigures = (whitePawns | whiteQueens);
	uint32_t blackFigures = (blackPawns | blackQueens);

	//left down beat
	uint32_t leftDownMove = SHIFTDOWN(mask, 3);
	uint32_t leftDownBeat = SHIFTDOWN(leftDownMove, 3);

	if (currentTurn == WHITE)
	{
		if (
			(mask & whitePawns) > 0 &&
			((mask | leftDownMove) & leftBorder) == 0 &&
			((mask | leftDownMove) & lowerBorder) == 0 &&
			(leftDownBeat & allFigures) == 0 &&
			(leftDownMove & blackFigures) > 0
			)
		{
			positions.push_back((mask | leftDownMove | leftDownBeat));
		}
	}
	if (currentTurn == BLACK)
	{
		if (
			(mask & blackPawns) > 0 &&
			((mask | leftDownMove) & leftBorder) == 0 &&
			((mask | leftDownMove) & lowerBorder) == 0 &&
			(leftDownBeat & allFigures) == 0 &&
			(leftDownMove & whiteFigures) > 0
			)
		{
			positions.push_back((mask | leftDownMove | leftDownBeat));
		}
	}


	//right down beat
	uint32_t rightDownMove = SHIFTDOWN(mask, 4);
	uint32_t rightDownBeat = SHIFTDOWN(rightDownMove, 4);

	if (currentTurn == WHITE)
	{
		if (
			(mask & whitePawns) > 0 &&
			((mask | rightDownMove) & rightBorder) == 0 &&
			((mask | rightDownMove) & lowerBorder) == 0 &&
			(rightDownBeat & allFigures) == 0 &&
			(rightDownMove & blackFigures) > 0
			)
		{
			positions.push_back((mask | rightDownMove | rightDownBeat));
		}
	}
	if (currentTurn == BLACK)
	{
		if (
			(mask & blackPawns) > 0 &&
			((mask | rightDownMove) & rightBorder) == 0 &&
			((mask | rightDownMove) & lowerBorder) == 0 &&
			(rightDownBeat & allFigures) == 0 &&
			(rightDownMove & whiteFigures) > 0
			)
		{
			positions.push_back((mask | rightDownMove | rightDownBeat));
		}
	}


	//left up beat
	uint32_t leftUpMove = SHIFTUP(mask, 4);
	uint32_t leftUpBeat = SHIFTUP(leftUpMove, 4);

	if (currentTurn == WHITE)
	{
		if (
			(mask & whitePawns) > 0 &&
			((mask | leftUpMove) & leftBorder) == 0 &&
			((mask | leftUpMove) & upperBorder) == 0 &&
			(leftUpBeat & allFigures) == 0 &&
			(leftUpMove & blackFigures) > 0
			)
		{
			positions.push_back((mask | leftUpMove | leftUpBeat));
		}
	}
	if (currentTurn == BLACK)
	{
		if (
			(mask & blackPawns) > 0 &&
			((mask | leftUpMove) & leftBorder) == 0 &&
			((mask | leftUpMove) & upperBorder) == 0 &&
			(leftUpBeat & allFigures) == 0 &&
			(leftUpMove & whiteFigures) > 0
			)
		{
			positions.push_back((mask | leftUpMove | leftUpBeat));
		}
	}


	//right up beat
	uint32_t rightUpMove = SHIFTUP(mask, 3);
	uint32_t rightUpBeat = SHIFTUP(rightUpMove, 3);

	if (currentTurn == WHITE)
	{
		if (
			(mask & whitePawns) > 0 &&
			((mask | rightUpMove) & rightBorder) == 0 &&
			((mask | rightUpMove) & upperBorder) == 0 &&
			(rightUpBeat & allFigures) == 0 &&
			(rightUpMove & blackFigures) > 0
			)
		{
			positions.push_back((mask | rightUpMove | rightUpBeat));
		}
	}
	if (currentTurn == BLACK)
	{
		if (
			(mask & blackPawns) > 0 &&
			((mask | rightUpMove) & rightBorder) == 0 &&
			((mask | rightUpMove) & upperBorder) == 0 &&
			(rightUpBeat & allFigures) == 0 &&
			(rightUpMove & whiteFigures) > 0
			)
		{
			positions.push_back((mask | rightUpMove | rightUpBeat));
		}
	}
}

__device__ __host__ DeviceVector<uint32_t> generateMoves(Board b)
{
	DeviceVector<uint32_t> positions{};
	uint32_t mask = 1;
	for (int i = 0; i < 32; i++)
	{
		generateFromFigure(mask, positions, b);
		mask = (mask << 1);
	}

	return positions;
}

std::ostream& operator<<(std::ostream& out, Board& board)
{
	uint32_t mask = 1;
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (i % 2 == 1)
			{
				out << ".";
			}

			if ((board.whitePawns & mask) > 0)
			{
				out << "p";
			}
			else if ((board.whiteQueens & mask) > 0)
			{
				out << "q";
			}
			else if ((board.blackPawns & mask) > 0)
			{
				out << "P";
			}
			else if ((board.blackQueens & mask) > 0)
			{
				out << "Q";
			}
			else
			{
				out << ".";
			}
			if (i % 2 == 0)
			{
				out << ".";
			}
			mask = (mask << 1);

		}
		out << std::endl;
	}
	return out;
}

__host__ bool isTheSameBoard(Board b1, Board b2)
{
	return b1.blackPawns == b2.blackPawns && b1.whitePawns == b2.whitePawns && b1.blackQueens == b2.blackQueens && b1.whiteQueens == b2.whiteQueens;
}

__device__ __host__ Board applyMove(Board board, uint32_t move)
{
	Board b;
	b.currentTurn = (!board.currentTurn);
	b.whitePawns = board.whitePawns;
	b.blackPawns = board.blackPawns;
	b.whiteQueens = board.whiteQueens;
	b.blackQueens = board.blackQueens;
	b.lastBeat = (board.lastBeat + 1);
	uint32_t allFigures = (board.blackPawns | board.blackQueens | board.whitePawns | board.whiteQueens);
	uint32_t finalMove = ((~allFigures) & move);
	if (board.currentTurn == WHITE)
	{
		if (move & board.whitePawns)
		{
			b.whitePawns = ((board.whitePawns & (~move)) | finalMove); 
		}
		if (move & board.whiteQueens)
		{
			b.whiteQueens = ((board.whiteQueens & (~move)) | finalMove);
		}
		b.blackPawns = (board.blackPawns & (~move));
		b.blackQueens = (board.blackQueens & (~move));

		if (b.blackPawns != board.blackPawns || b.blackQueens != board.blackQueens)
		{
			b.lastBeat = 0;
		}
	}
	else
	{
		if (move & board.blackPawns)
		{
			b.blackPawns = ((board.blackPawns & (~move)) | finalMove);
		}
		if (move & board.blackQueens)
		{
			b.blackQueens = ((board.blackQueens & (~move)) | finalMove);
		}
		b.whitePawns = (board.whitePawns & (~move));
		b.whiteQueens = (board.whiteQueens & (~move));

		if (b.whitePawns != board.whitePawns || b.whiteQueens != board.whiteQueens)
		{
			b.lastBeat = 0;
		}
	}
	b.whiteQueens = (b.whiteQueens | (b.whitePawns & lowerBorder));
	b.blackQueens = (b.blackQueens | (b.blackPawns & upperBorder));
	b.whitePawns = (b.whitePawns & (~lowerBorder));
	b.blackPawns = (b.blackPawns & (~upperBorder));

	return b;
}

//__device__ __host__ uint32_t shift(uint32_t mask, int amount)
//{
//	return (mask & additionalShiftMask) > 0 ? (mask << (amount + 1)) : (mask << amount);
//}
