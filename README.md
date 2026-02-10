# ML Chess Bot 🎮♟️🤖

A terminal-based chess bot using **Random Forest Classification** to learn from historical chess games and play against you.

## Features

- **True Machine Learning**: Random Forest classifier trained on historical positions
- **Feature Engineering**: 152 numerical features extracted from each board position
- **Memory Optimized**: Trains on 100K positions from 2,000 games
- **Smart Prediction**: Probability-based move selection with legal move filtering
- **Interactive Terminal**: Play directly in terminal with algebraic notation

## Quick Start

1. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn python-chess
```

2. **Run the bot**:
```bash
python main.py
```

## How It Works

### Machine Learning Model
- **Algorithm**: Random Forest (50 trees, max depth 15)
- **Training Data**: ~100,000 position-move pairs from 2,000 games
- **Features**: 152 numerical features per position including:
  - Material count (12 features)
  - Piece distribution (16 features)  
  - Center control (2 features)
  - Game state (8 features: turn, check, castling rights, mobility)

### Gameplay
1. Choose bot color (White or Black)
2. Enter moves in algebraic notation (e4, Nf3, Qxd5, etc.)
3. Bot predicts moves using Random Forest model
4. Falls back to heuristics for unseen positions

## Files

- **model.py**: ChessBot class with ML model and feature extraction
- **main.py**: Game loop and user interface
- **games.csv**: Training dataset (20,000+ chess games)

## Model Performance

- **Training**: ~30-60 seconds on first run
- **Memory**: ~500-800 MB during training
- **Move Vocabulary**: ~1,800 unique moves learned
- **Inference**: Instant predictions during gameplay

## Commands

- `quit` - Exit the game
- `resign` - Resign the current game

## Example

```
Choose bot color:
1. White
2. Black

Your choice: 2
Bot will play as BLACK

Your move: e4
Bot plays: e5

Your move: Nf3  
Bot plays: Nc6
```

## License

MIT License - Feel free to use and modify!

## Author

Built with ♟️ and 🤖
