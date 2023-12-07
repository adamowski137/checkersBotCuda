#pragma once
#include <SDL.h>
#include <SDL_image.h>
#include <condition_variable>

#include "Board.cuh"

class Window
{
public:
	Window(std::condition_variable& cv);
	~Window();

	void runWindow();
	void updateBoard(Board board);
	Board boardUpdated();
	void finishGame();
	
private:

	const int HEIGHT;
	const int WIDTH;

	Board board;

	uint32_t selectedFigure;
	std::condition_variable& cv;

	SDL_Window* window;
	SDL_Renderer* renderer;
	SDL_Texture* boardTexture;
	SDL_Texture* figuresTexture;
	SDL_Texture* availableMovesTexture;

	SDL_Texture* specialTile;
	SDL_Texture* blackPawn;
	SDL_Texture* whitePawn;
	SDL_Texture* blackQueen;
	SDL_Texture* whiteQueen;

	SDL_mutex* mutex;

	void handleClick(int x, int y);
	bool quit;
};