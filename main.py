import chess
from model import ChessBot


def display_board(board, bot_color):
    """Display the chess board in the terminal"""
    print("\n" + "="*40)
    if bot_color == chess.WHITE:
        print(board)
    else:
        # Flip board for black
        print(board.transform(chess.flip_vertical).transform(chess.flip_horizontal))
    print("="*40)


def play_game():
    """Main game loop"""
    print("="*50)
    print("   CHESS BOT - Machine Learning Powered")
    print("   Using Random Forest Classification")
    print("   (Memory Optimized)")
    print("="*50)
    print()
    
    # Initialize bot
    bot = ChessBot()
    
    # Try to load existing model
    if not bot.load_model():
        print("No trained model found. Training new ML model...")
        print("(Memory-optimized: ~2000 games, ~100k positions)")
        print("This will take 30-60 seconds...")
        bot.load_and_train()
        bot.save_model()
    
    # Choose bot color
    print("\nChoose bot color:")
    print("1. White")
    print("2. Black")
    
    while True:
        choice = input("\nYour choice (1 or 2): ").strip()
        if choice == '1':
            bot_color = chess.WHITE
            player_color = chess.BLACK
            print("\nBot will play as WHITE")
            break
        elif choice == '2':
            bot_color = chess.BLACK
            player_color = chess.WHITE
            print("\nBot will play as BLACK")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Initialize board
    board = chess.Board()
    
    print("\n" + "="*50)
    print("Game started! Enter moves in algebraic notation (e.g., e4, Nf3)")
    print("Type 'quit' to exit, 'resign' to resign")
    print("="*50)
    
    # Game loop
    while not board.is_game_over():
        display_board(board, bot_color)
        
        # Show game status
        if board.is_check():
            print(f"\n{'WHITE' if board.turn == chess.WHITE else 'BLACK'} is in CHECK!")
        
        print(f"\nMove {board.fullmove_number}: {'White' if board.turn == chess.WHITE else 'Black'} to move")
        
        # Determine whose turn it is
        if board.turn == bot_color:
            # Bot's turn
            print("Bot is thinking... (using ML model)")
            bot_move = bot.get_bot_move(board)
            
            if bot_move is None:
                print("Bot has no legal moves!")
                break
            
            print(f"Bot plays: {board.san(bot_move)}")
            board.push(bot_move)
            
        else:
            # Player's turn
            print("\nLegal moves:", ", ".join([board.san(m) for m in board.legal_moves]))
            
            while True:
                move_input = input("\nYour move: ").strip()
                
                if move_input.lower() == 'quit':
                    print("Game ended by player.")
                    return
                
                if move_input.lower() == 'resign':
                    print(f"\n{'Black' if player_color == chess.BLACK else 'White'} resigns. Bot wins!")
                    return
                
                try:
                    # Try to parse the move
                    move = board.parse_san(move_input)
                    board.push(move)
                    break
                except ValueError:
                    print("Invalid move! Please enter a legal move in algebraic notation.")
    
    # Game over
    display_board(board, bot_color)
    print("\n" + "="*50)
    print("GAME OVER!")
    print("="*50)
    
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate! Game is a draw.")
    elif board.is_insufficient_material():
        print("Draw due to insufficient material.")
    elif board.is_fifty_moves():
        print("Draw by fifty-move rule.")
    elif board.is_repetition():
        print("Draw by threefold repetition.")
    else:
        print("Game ended.")
    
    print(f"\nFinal position FEN: {board.fen()}")


if __name__ == "__main__":
    play_game()
