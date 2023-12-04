#pragma once

#include "Board.cuh"


__device__ __host__ Board::Board()
{
	whitePawns = 0x55AA55;
	blackPawns = 0xAA55AA0000000000;
	whiteQueens = 0;
	blackQueens = 0;
	currentTurn = WHITE;
}

__device__ __host__ Board::Board(uint64_t whitePawns, uint64_t blackPawns, uint64_t whiteQueens, uint64_t blackQueens, bool currentTurn)
{
	this->whitePawns = whitePawns;
	this->blackPawns = blackPawns;
	this->whiteQueens = whiteQueens;
	this->blackQueens = blackQueens;
	this->currentTurn = currentTurn;
}


__device__ __host__ DeviceVector<Board> Board::generatePositions()
{
	DeviceVector<Board> positions{};
	uint64_t mask = 1;
	for (int i = 0; i < 64; i++)
	{
		generateFromFigure(mask, positions);
		mask = (mask << 1);
	}

	return positions;
}


__device__ __host__ bool Board::isTerminal(bool* winner)
{
	if ((whitePawns | whiteQueens) == 0)
	{
		*winner = BLACK;
		return true;
	}

	if ((blackPawns | blackQueens) == 0)
	{
		*winner = WHITE;
		return true;
	}
	return false;
}

__device__ __host__ void Board::promoteToQueen(uint64_t& pawns, uint64_t& queens, uint64_t mask)
{
	if (currentTurn == WHITE)
	{
		if ((mask & lowerBorder) > 0)
		{
			pawns = (pawns & (~mask));
			queens = (queens | mask);
		}
	}
	if (currentTurn == BLACK)
	{
		if ((mask & upperBorder) > 0)
		{
			pawns = (pawns & (~mask));
			queens = (queens | mask);
		}
	}
}

__device__ __host__ void Board::generateQueenMoves(uint64_t mask, DeviceVector<Board>& positions)
{
	uint64_t allFigures = (whitePawns | blackPawns | whiteQueens | blackQueens);
	uint64_t whiteFigures = (whitePawns | whiteQueens);
	uint64_t blackFigures = (blackPawns | blackQueens);


	if (currentTurn == WHITE)
	{
		if ((mask & leftBorder) == 0)
		{
			uint64_t leftUp = (mask >> 9);
			uint64_t current = mask;

			while (leftUp > 0 && (leftUp & whiteFigures) == 0 && (current & leftBorder) == 0)
			{
				if ((leftUp & allFigures) > 0)
				{
					if ((leftUp & blackFigures) > 0 && (leftUp & leftBorder) == 0)
					{
						uint64_t leftBeat = (leftUp >> 9);
						if (leftBeat > 0 && (leftBeat & allFigures) == 0)
						{
							uint64_t newWhiteQueens = ((whiteQueens | leftBeat) & (~mask));
							uint64_t newBlackPawns = ((blackPawns) & (~leftUp));
							uint64_t newBlackQueens = ((blackQueens) & (~leftUp));
							uint64_t newWhitePawns = whitePawns;

							positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
						}
					}
					break;
				}

				uint64_t newWhiteQueens = ((whiteQueens | leftUp) & (~mask));
				positions.push_back(Board(whitePawns, blackPawns, newWhiteQueens, blackQueens, !currentTurn));
				current = leftUp;
				leftUp = (leftUp >> 9);
			}
		}
		if ((mask & rightBorder) == 0)
		{
			uint64_t rightUp = (mask >> 7);
			uint64_t current = mask;

			while (rightUp > 0 && (rightUp & whiteFigures) == 0 && (current & rightBorder) == 0)
			{
				if ((rightUp & allFigures) > 0)
				{
					if ((rightUp & blackFigures) > 0 && (rightUp & rightBorder) == 0)
					{
						uint64_t rightBeat = (rightUp >> 7);
						if (rightBeat > 0 && (rightBeat & allFigures) == 0)
						{
							uint64_t newWhiteQueens = ((whiteQueens | rightBeat) & (~mask));
							uint64_t newBlackPawns = ((blackPawns) & (~rightUp));
							uint64_t newBlackQueens = ((blackQueens) & (~rightUp));
							uint64_t newWhitePawns = whitePawns;

							positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
						}
					}
					break;
				}

				uint64_t newWhiteQueens = ((whiteQueens | rightUp) & (~mask));
				positions.push_back(Board(whitePawns, blackPawns, newWhiteQueens, blackQueens, !currentTurn));

				current = rightUp;
				rightUp = (rightUp >> 7);
			}
		}
		if ((mask & leftBorder) == 0)
		{
			uint64_t leftDown = (mask << 7);
			uint64_t current = mask;

			while (leftDown > 0 && (leftDown & whiteFigures) == 0 && (current & leftBorder) == 0)
			{
				if ((leftDown & allFigures) > 0)
				{
					if ((leftDown & blackFigures) > 0 && (leftDown & leftBorder) == 0)
					{
						uint64_t leftDownBeat = (leftDown << 7);
						if (leftDownBeat > 0 && (leftDownBeat & allFigures) == 0)
						{
							uint64_t newWhiteQueens = ((whiteQueens | leftDownBeat) & (~mask));
							uint64_t newBlackPawns = ((blackPawns) & (~leftDown));
							uint64_t newBlackQueens = ((blackQueens) & (~leftDown));
							uint64_t newWhitePawns = whitePawns;

							positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));

						}
					}
					break;
				}


				uint64_t newWhiteQueens = ((whiteQueens | leftDown) & (~mask));
				positions.push_back(Board(whitePawns, blackPawns, newWhiteQueens, blackQueens, !currentTurn));
				current = leftDown;
				leftDown = (leftDown << 7);
			}
		}

		if ((mask & rightBorder) == 0)
		{
			uint64_t current = mask;
			uint64_t rightDown = (mask << 9);
			while (rightDown > 0 && (rightDown & whiteFigures) == 0 && (current & rightBorder) == 0)
			{
				if ((rightDown & allFigures) > 0)
				{
					if ((rightDown & blackFigures) > 0 && (rightDown & rightBorder) == 0)
					{
						uint64_t rightDownBeat = (rightDown << 9);
						if (rightDownBeat > 0 && (rightDownBeat & allFigures) == 0)
						{
							uint64_t newWhiteQueens = ((whiteQueens | rightDownBeat) & (~mask));
							uint64_t newBlackPawns = ((blackPawns) & (~rightDown));
							uint64_t newBlackQueens = ((blackQueens) & (~rightDown));
							uint64_t newWhitePawns = whitePawns;

							promoteToQueen(newWhitePawns, newWhiteQueens, rightDownBeat);

							positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
						}
					}
					break;
				}

				uint64_t newWhiteQueens = ((whiteQueens | rightDown) & (~mask));
				positions.push_back(Board(whitePawns, blackPawns, newWhiteQueens, blackQueens, !currentTurn));
				current = rightDown;
				rightDown = (rightDown << 9);
			}
		}
	}
	if (currentTurn == BLACK)
	{
		if ((mask & leftBorder) == 0)
		{
			uint64_t leftUp = (mask >> 9);
			uint64_t current = mask;

			while (leftUp > 0 && (leftUp & blackFigures) == 0 && (current & leftBorder) == 0)
			{
				if ((leftUp & allFigures) > 0)
				{
					if ((leftUp & whiteFigures) > 0 && (leftUp & leftBorder) == 0)
					{
						uint64_t leftBeat = (leftUp >> 9);
						if (leftBeat > 0 && (leftBeat & allFigures) == 0)
						{
							uint64_t newBlackQueens = ((blackQueens | leftBeat) & (~mask));
							uint64_t newWhitePawns = ((whitePawns) & (~leftUp));
							uint64_t newWhiteQueens = ((whiteQueens) & (~leftUp));
							uint64_t newBlackPawns = blackPawns;

							positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
						}
					}
					break;
				}

				uint64_t newBlackQueens = ((blackQueens | leftUp) & (~mask));
				positions.push_back(Board(whitePawns, blackPawns, whiteQueens, newBlackQueens, !currentTurn));
				current = leftUp;

				leftUp = (leftUp >> 9);
			}
		}

		if ((mask & rightBorder) == 0)
		{
			uint64_t rightUp = (mask >> 7);
			uint64_t current = mask;

			while (rightUp > 0 && (rightUp & blackFigures) == 0 && (current & rightBorder) == 0)
			{
				if ((rightUp & allFigures) > 0)
				{
					if ((rightUp & whiteFigures) > 0 && (rightUp & rightBorder) == 0)
					{
						uint64_t rightBeat = (rightUp >> 7);
						if (rightBeat > 0 && (rightBeat & allFigures) == 0)
						{
							uint64_t newBlackQueens = ((blackQueens | rightBeat) & (~mask));
							uint64_t newWhitePawns = ((whitePawns) & (~rightUp));
							uint64_t newWhiteQueens = ((whiteQueens) & (~rightUp));
							uint64_t newBlackPawns = blackPawns;

							positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));

						}
					}
					break;
				}

				uint64_t newBlackQueens = ((blackQueens | rightUp) & (~mask));
				positions.push_back(Board(whitePawns, blackPawns, whiteQueens, newBlackQueens, !currentTurn));
				current = rightUp;
				rightUp = (rightUp >> 7);
			}
		}

		if ((mask & leftBorder) == 0)
		{
			uint64_t leftDown = (mask << 7);
			uint64_t current = mask;

			while (leftDown > 0 && (leftDown & blackFigures) == 0 && (current & leftBorder) == 0)
			{
				if ((leftDown & allFigures) > 0)
				{
					if ((leftDown & whiteFigures) > 0 && (leftDown & leftBorder) == 0)
					{
						uint64_t leftDownBeat = (leftDown << 7);
						if (leftDownBeat > 0 && (leftDownBeat & allFigures) == 0)
						{
							uint64_t newBlackQueens = ((blackQueens | leftDownBeat) & (~mask));
							uint64_t newWhitePawns = ((whitePawns) & (~leftDown));
							uint64_t newWhiteQueens = ((whiteQueens) & (~leftDown));
							uint64_t newBlackPawns = blackPawns;

							positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));

						}
					}
					break;
				}

				uint64_t newBlackQueens = ((blackQueens | leftDown) & (~mask));
				positions.push_back(Board(whitePawns, blackPawns, whiteQueens, newBlackQueens, !currentTurn));
				current = leftDown;

				leftDown = (leftDown << 7);
			}
		}

		if ((mask & rightBorder) == 0)
		{
			uint64_t current = mask;
			uint64_t rightDown = (mask << 9);
			while (rightDown > 0 && (rightDown & blackFigures) == 0 && (current & rightBorder) == 0)
			{
				if ((rightDown & allFigures) > 0)
				{

					if ((rightDown & whiteFigures) > 0 && (rightDown & rightBorder) == 0)
					{
						uint64_t rightDownBeat = (rightDown << 9);
						if (rightDownBeat > 0 && (rightDownBeat & allFigures) == 0)
						{
							uint64_t newBlackQueens = ((blackQueens | rightDownBeat) & (~mask));
							uint64_t newWhitePawns = ((whitePawns) & (~rightDown));
							uint64_t newWhiteQueens = ((whiteQueens) & (~rightDown));
							uint64_t newBlackPawns = blackPawns;

							positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
						}
					}
					break;
				}


				uint64_t newBlackQueens = ((blackQueens | rightDown) & (~mask));
				positions.push_back(Board(whitePawns, blackPawns, whiteQueens, newBlackQueens, !currentTurn));
				current = rightDown;
				rightDown = (rightDown << 9);
			}
		}
	}
}
__device__ __host__ Board& Board::operator=(const Board& other) {
	if (this != &other) {
		// Copy the data members from 'other' to 'this'
		this->whitePawns = other.whitePawns;
		this->whiteQueens = other.whiteQueens;
		this->blackPawns = other.blackPawns;
		this->blackQueens = other.blackQueens;
		this->currentTurn = other.currentTurn;
	}
	return *this;
}

__device__ __host__ DeviceVector<Board> Board::generateFromFigure(uint64_t mask, DeviceVector<Board>& positions)
{
	uint64_t allFigures = (whitePawns | blackPawns | whiteQueens | blackQueens);
	uint64_t whiteFigures = (whitePawns | whiteQueens);
	uint64_t blackFigures = (blackPawns | blackQueens);


	

	generatePawnBeatMoves(mask, positions);
	
	if (currentTurn == WHITE)
	{
		if (mask & whiteQueens)
		{
			generateQueenMoves(mask, positions);
			return positions;
		}
		if ((mask & whitePawns) > 0)
		{
			uint64_t leftMove = (mask << 7);

			if (leftMove > 0 && (mask & leftBorder) == 0)
			{
				if ((leftMove & allFigures) == 0)
				{
					uint64_t newWhitePawns = ((whitePawns | leftMove) & (~mask));
					uint64_t newWhiteQueens = whiteQueens;

					promoteToQueen(newWhitePawns, newWhiteQueens, leftMove);

					positions.push_back(Board(newWhitePawns, blackPawns, newWhiteQueens, blackQueens, !currentTurn));
				}

			}
			uint64_t rightMove = (mask << 9);

			if (rightMove > 0 && (mask & rightBorder) == 0)
			{
				if ((rightMove & allFigures) == 0)
				{
					uint64_t newWhitePawns = ((whitePawns | rightMove) & (~mask));
					uint64_t newWhiteQueens = whiteQueens;

					promoteToQueen(newWhitePawns, newWhiteQueens, rightMove);

					positions.push_back(Board(newWhitePawns, blackPawns, newWhiteQueens, blackQueens, !currentTurn));
				}
			}
		}
	}
	if (currentTurn == BLACK)
	{
		if (mask & blackQueens)
		{
			generateQueenMoves(mask, positions);
			return positions;
		}
		if ((mask & blackPawns) > 0)
		{
			uint64_t leftMove = (mask >> 9);

			if (leftMove > 0 && (mask & leftBorder) == 0)
			{
				if ((leftMove & allFigures) == 0)
				{
					uint64_t newBlackPawns = ((blackPawns | leftMove) & (~mask));
					uint64_t newBlackQueens = blackQueens;

					promoteToQueen(newBlackPawns, newBlackQueens, leftMove);

					positions.push_back(Board(whitePawns, newBlackPawns, whiteQueens, newBlackQueens, !currentTurn));
				}

			}
			uint64_t rightMove = (mask >> 7);

			if (rightMove > 0 && (mask & rightBorder) == 0)
			{
				if ((rightMove & allFigures) == 0)
				{
					uint64_t newBlackPawns = ((blackPawns | rightMove) & (~mask));
					uint64_t newBlackQueens = blackQueens;

					promoteToQueen(newBlackPawns, newBlackQueens, rightMove);

					positions.push_back(Board(whitePawns, newBlackPawns, whiteQueens, newBlackQueens, !currentTurn));
				}
			}
		}
	}



	return positions;
}

__device__ __host__ bool Board::operator==(const Board& other)
{
	return blackPawns == other.blackPawns && whitePawns == other.whitePawns && blackQueens == other.blackQueens && whiteQueens == other.whiteQueens;
}

void Board::generatePawnBeatMoves(uint64_t mask, DeviceVector<Board>& positions)
{
	uint64_t allFigures = (whitePawns | blackPawns | whiteQueens | blackQueens);
	uint64_t whiteFigures = (whitePawns | whiteQueens);
	uint64_t blackFigures = (blackPawns | blackQueens);

	//left down beat
	uint64_t leftDownMove = (mask << 7);
	uint64_t leftDownBeat = (leftDownMove << 7);

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
			uint64_t newWhitePawns = ((whitePawns | leftDownBeat) & (~mask));
			uint64_t newBlackPawns = ((blackPawns) & (~leftDownMove));
			uint64_t newBlackQueens = ((blackQueens) & (~leftDownMove));
			uint64_t newWhiteQueens = whiteQueens;

			promoteToQueen(newWhitePawns, newWhiteQueens, leftDownBeat);

			positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
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
			uint64_t newBlackPawns = ((blackPawns | leftDownBeat) & (~mask));
			uint64_t newWhitePawns = ((whitePawns) & (~leftDownMove));
			uint64_t newWhiteQueens = ((whiteQueens) & (~leftDownMove));
			uint64_t newBlackQueens = blackQueens;

			promoteToQueen(newBlackPawns, newBlackQueens, leftDownBeat);

			positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
		}
	}


	//right down beat
	uint64_t rightDownMove = (mask << 9);
	uint64_t rightDownBeat = (rightDownMove << 9);

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
			uint64_t newWhitePawns = ((whitePawns | rightDownBeat) & (~mask));
			uint64_t newBlackPawns = ((blackPawns) & (~rightDownMove));
			uint64_t newBlackQueens = ((blackQueens) & (~rightDownMove));
			uint64_t newWhiteQueens = whiteQueens;

			promoteToQueen(newWhitePawns, newWhiteQueens, rightDownBeat);

			positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
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
			uint64_t newBlackPawns = ((blackPawns | rightDownBeat) & (~mask));
			uint64_t newWhitePawns = ((whitePawns) & (~rightDownMove));
			uint64_t newWhiteQueens = ((whiteQueens) & (~rightDownMove));
			uint64_t newBlackQueens = blackQueens;

			promoteToQueen(newBlackPawns, newBlackQueens, rightDownBeat);

			positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
		}
	}


	//left up beat
	uint64_t leftUpMove = (mask >> 9);
	uint64_t leftUpBeat = (leftUpMove >> 9);

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
			uint64_t newWhitePawns = ((whitePawns | leftUpBeat) & (~mask));
			uint64_t newBlackPawns = ((blackPawns) & (~leftUpMove));
			uint64_t newBlackQueens = ((blackQueens) & (~leftUpMove));
			uint64_t newWhiteQueens = whiteQueens;

			promoteToQueen(newWhitePawns, newWhiteQueens, leftUpBeat);

			positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
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
			uint64_t newBlackPawns = ((blackPawns | leftUpBeat) & (~mask));
			uint64_t newWhitePawns = ((whitePawns) & (~leftUpMove));
			uint64_t newWhiteQueens = ((whiteQueens) & (~leftUpMove));
			uint64_t newBlackQueens = blackQueens;

			promoteToQueen(newBlackPawns, newBlackQueens, leftUpBeat);

			positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
		}
	}


	//right up beat
	uint64_t rightUpMove = (mask >> 7);
	uint64_t rightUpBeat = (rightUpMove >> 7);

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
			uint64_t newWhitePawns = ((whitePawns | rightUpBeat) & (~mask));
			uint64_t newBlackPawns = ((blackPawns) & (~rightUpMove));
			uint64_t newBlackQueens = ((blackQueens) & (~rightUpMove));
			uint64_t newWhiteQueens = whiteQueens;

			promoteToQueen(newWhitePawns, newWhiteQueens, rightUpBeat);

			positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
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
			uint64_t newBlackPawns = ((blackPawns | rightUpBeat) & (~mask));
			uint64_t newWhitePawns = ((whitePawns) & (~rightUpMove));
			uint64_t newWhiteQueens = ((whiteQueens) & (~rightUpMove));
			uint64_t newBlackQueens = blackQueens;

			promoteToQueen(newBlackPawns, newBlackQueens, rightUpBeat);

			positions.push_back(Board(newWhitePawns, newBlackPawns, newWhiteQueens, newBlackQueens, !currentTurn));
		}
	}
}



std::ostream& operator<<(std::ostream& out, Board& board)
{
	uint64_t mask = 1;
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
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
			else {
				out << ".";
			}
			mask = (mask << 1);

		}
		out << std::endl;
	}
	return out;
}
