#include "Window.hpp"
#include <iostream>

Window::Window(std::condition_variable& cv) : HEIGHT{ 800 }, WIDTH{ 800 }, cv{cv}
{
    board = createBoard();
    quit = false;
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        throw("failed to initialize window");
        return;
    }

    window = SDL_CreateWindow("Checkers Bot", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    if (window == NULL)
    {
        throw("failed to create window");
        return;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == NULL)
    {
        throw("failed to create renderer");
        return;
    }

    figuresTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, WIDTH, HEIGHT);
    availableMovesTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, WIDTH, HEIGHT);
    specialTile = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, WIDTH, HEIGHT);




    SDL_Surface* tmp = IMG_Load("Chessboard.png");
    if (tmp == NULL)
    {
        std::cout << "Can't load";
    }

    boardTexture = SDL_CreateTextureFromSurface(renderer, tmp);

    tmp = IMG_Load("blackPawn.png");
    blackPawn = SDL_CreateTextureFromSurface(renderer, tmp);

    tmp = IMG_Load("whitePawn.png");
    whitePawn = SDL_CreateTextureFromSurface(renderer, tmp);

    tmp = IMG_Load("blackQueen.png");
    blackQueen = SDL_CreateTextureFromSurface(renderer, tmp);

    tmp = IMG_Load("whiteQueen.png");
    whiteQueen = SDL_CreateTextureFromSurface(renderer, tmp);

    SDL_FreeSurface(tmp);

    mutex = SDL_CreateMutex();

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    SDL_SetTextureBlendMode(blackPawn, SDL_BLENDMODE_BLEND);
    SDL_SetTextureBlendMode(whitePawn, SDL_BLENDMODE_BLEND);
    SDL_SetTextureBlendMode(blackQueen, SDL_BLENDMODE_BLEND);
    SDL_SetTextureBlendMode(whiteQueen, SDL_BLENDMODE_BLEND);

    SDL_SetTextureBlendMode(boardTexture, SDL_BLENDMODE_BLEND);
    SDL_SetTextureBlendMode(figuresTexture, SDL_BLENDMODE_BLEND);
    SDL_SetTextureBlendMode(availableMovesTexture, SDL_BLENDMODE_BLEND);
    SDL_SetTextureBlendMode(specialTile, SDL_BLENDMODE_BLEND);

    SDL_SetRenderTarget(renderer, specialTile);
    SDL_SetRenderDrawColor(renderer, 255, 255, 0, 150);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);

    selectedFigure = 0;
}

Window::~Window()
{
    SDL_DestroyTexture(boardTexture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

Board Window::boardUpdated()
{
    return board;
}

void Window::finishGame()
{
    quit = true;
}

void Window::runWindow()
{
    SDL_Event e;

    while (!quit)
    {
        while (SDL_PollEvent(&e) != 0)
        {
            if (e.type == SDL_QUIT)
            {
                quit = true;
            }
            if (e.type == SDL_MOUSEBUTTONDOWN)
            {
                int x = e.button.x;
                int y = e.button.y;
                handleClick(x, y);
            }
        }
        SDL_LockMutex(mutex);
        SDL_SetRenderTarget(renderer, NULL);

        SDL_RenderCopy(renderer, boardTexture, NULL, NULL);
        SDL_RenderCopy(renderer, figuresTexture, NULL, NULL);
        SDL_RenderCopy(renderer, availableMovesTexture, NULL, NULL);

        SDL_RenderPresent(renderer);
        SDL_UnlockMutex(mutex);
    }
}

void Window::updateBoard(Board board)
{
    this->board = board;
    uint64_t mask = 1;
    int width = WIDTH / 8;
    int height = HEIGHT / 8;
    SDL_LockMutex(mutex);
    SDL_SetRenderTarget(renderer, figuresTexture);
    SDL_RenderClear(renderer);
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            SDL_Rect rect;
            rect.h = height;
            rect.w = width;
            rect.x = width * (j * 2 + (i % 2));
            rect.y = height * (i);
            if ((mask & board.blackPawns) > 0)
            {
                SDL_RenderCopy(renderer, blackPawn, NULL, &rect);
            }
            if ((mask & board.whitePawns) > 0)
            {
                SDL_RenderCopy(renderer, whitePawn, NULL, &rect);
            }
            if ((mask & board.blackQueens) > 0)
            {
                SDL_RenderCopy(renderer, blackQueen, NULL, &rect);
            }
            if ((mask & board.whiteQueens) > 0)
            {
                SDL_RenderCopy(renderer, whiteQueen, NULL, &rect);
            }
            mask = (mask << 1);            
        }
    }
    SDL_UnlockMutex(mutex);
}


void Window::handleClick(int x, int y)
{
    int squareHeight = HEIGHT / 8;
    int squareWidth = WIDTH / 8;

    SDL_SetRenderTarget(renderer, availableMovesTexture);
    SDL_RenderClear(renderer);

    int selectedY = (y / squareHeight);
    int selectedX = (x / squareWidth);
    if ((selectedX + selectedY) % 2 == 1)
    {
        selectedFigure = 0;
        return;
    }

    uint32_t mask = ((uint32_t)1<< (selectedY * 4 + selectedX / 2));

    if(selectedFigure == 0)
    {
        if (board.currentTurn == WHITE)
        {
            if (((board.whitePawns | board.whiteQueens) & mask) > 0)
            {
                selectedFigure = mask;
            }
        }
        if (board.currentTurn == BLACK)
        {
            if (((board.blackPawns | board.blackQueens) & mask) > 0)
            {
                selectedFigure = mask;
            }
        }

        if (selectedFigure != 0)
        {
            SDL_Rect rect;
            rect.x = (x / squareWidth) * squareWidth;
            rect.y = (y / squareHeight) * squareHeight;
            rect.w = squareWidth;
            rect.h = squareHeight;
            SDL_RenderCopy(renderer, specialTile, NULL, &rect);
            DeviceVector<uint32_t> moves;
            generateFromFigure(mask, moves, board);

            uint32_t availableMask = 0;
            for (int i = 0; i < moves.size(); i++)
            {
                availableMask = (availableMask | moves[i]);
            }

            uint32_t current = (board.blackPawns | board.whitePawns | board.blackQueens | board.whiteQueens);

            availableMask = (availableMask & (~current));

            uint32_t tmp = 1;
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    if ((availableMask & tmp) > 0)
                    {
                        SDL_Rect rect;
                        rect.h = squareHeight;
                        rect.w = squareWidth;
                        rect.x = squareWidth * (j * 2 + (i % 2));
                        rect.y = squareHeight * (i);
                        SDL_RenderCopy(renderer, specialTile, NULL, &rect);
                    }
                    tmp = (tmp << 1);
                }
            }

            return;
        }
    }
    else 
    {

        DeviceVector<uint32_t> moves;
        uint32_t current = (board.blackPawns | board.whitePawns | board.blackQueens | board.whiteQueens);

        generateFromFigure(selectedFigure, moves, board);
        for (int i = 0; i < moves.size(); i++)
        {
            if ((moves[i] & (~current) & mask) > 0)
            {
                updateBoard(applyMove(board, moves[i]));
                cv.notify_all();
                break;
            }
        }

        selectedFigure = 0;

    }
}
